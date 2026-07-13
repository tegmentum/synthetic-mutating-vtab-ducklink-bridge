//! Synthetic ducklink writer bridge (test-only).
//!
//! Exports `duckdb:extension/storage-write-dispatch@2.0.0` -- the WRITABLE half
//! of the DuckDB storage-backend contract (transactions + DDL + DML). Companion
//! to the sqlink `synthetic-mutating-vtab-bridge`: the SDK-side boundary test
//! at `ducklink/crates/storage-boundary-test/tests/write_boundary.rs` loads
//! this component and drives ducklink-runtime's write trampolines end-to-end
//! against it, validating the runtime -> component wire without depending on
//! the DuckDB C++ `StorageExtension` shim (which is out of scope for this SDK).
//!
//! Semantics: a synthetic `kv_store(key TEXT, value TEXT)` backend. State is
//! a process-local `Mutex<KvState>` holding:
//!   * `committed`: the last-committed BTreeMap<rowid, (key, value)>.
//!   * `shadow`: the current transaction's mutable snapshot, promoted on
//!     commit / discarded on rollback.
//! No savepoints (DuckDB's TransactionManager doesn't use SAVEPOINT the same
//! way SQLite's vtab-update does).
//!
//! `handle` and `catalog` / `txn` handles are honored as-is; the bridge accepts
//! any `handle` value (it does not model a callback registry -- there's exactly
//! one `kv_store`). `catalog` is validated only when a transaction opens; `txn`
//! is a monotonic counter allocated at begin.
#![allow(unsafe_op_in_unsafe_fn)]

mod bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "writer",
        generate_all,
    });
}

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use bindings::exports::duckdb::extension::storage_write_dispatch::Guest;
use bindings::duckdb::extension::types::{Columndef, Duckerror, Duckvalue};

// -----------------------------------------------------------
// Backing store: rowid -> (key, value).
//
// A single global store keyed on rowid mirrors what a persistent
// storage backend would look like without a real disk. The txn
// state (`shadow`) captures the pre-txn snapshot so rollback can
// restore it; commit promotes shadow -> committed.
// -----------------------------------------------------------

const CATALOG_HANDLE: u32 = 1;
const TABLE_NAME: &str = "kv_store";

#[derive(Default, Clone)]
struct KvRows {
    rows: BTreeMap<i64, (String, String)>,
    next_rowid: i64,
}

#[derive(Default)]
struct KvState {
    committed: KvRows,
    /// Present between `begin` and `commit` / `rollback`. Holds the
    /// pre-txn snapshot so rollback can restore it.
    shadow: Option<KvRows>,
    /// Monotonic txn counter. Handed out by `begin-transaction` and
    /// checked by every subsequent write call.
    next_txn: u32,
    /// The currently-open txn handle, if any. DuckDB's TransactionManager
    /// serializes writes per catalog, so at most one live txn at a time.
    active_txn: Option<u32>,
    /// True iff `create-table("kv_store", ...)` has been called within
    /// the current or any prior txn. The schema is fixed to
    /// (key TEXT, value TEXT).
    created: bool,
}

fn state() -> &'static Mutex<KvState> {
    static S: OnceLock<Mutex<KvState>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(KvState::default()))
}

fn poison() -> Duckerror {
    Duckerror::Internal("synthetic-writer: state mutex poisoned".to_string())
}

fn require_catalog(catalog: u32) -> Result<(), Duckerror> {
    if catalog == CATALOG_HANDLE {
        Ok(())
    } else {
        Err(Duckerror::Invalidargument(format!(
            "synthetic-writer: unknown catalog handle {catalog} (only {CATALOG_HANDLE} is served)"
        )))
    }
}

fn require_txn(st: &KvState, txn: u32) -> Result<(), Duckerror> {
    match st.active_txn {
        Some(t) if t == txn => Ok(()),
        Some(other) => Err(Duckerror::Invalidstate(format!(
            "synthetic-writer: txn {txn} not active (current active txn is {other})"
        ))),
        None => Err(Duckerror::Invalidstate(format!(
            "synthetic-writer: txn {txn} not active (no live transaction)"
        ))),
    }
}

fn require_table(st: &KvState, table: &str) -> Result<(), Duckerror> {
    if !st.created {
        return Err(Duckerror::Invalidstate(format!(
            "synthetic-writer: table '{table}' has not been created"
        )));
    }
    if table != TABLE_NAME {
        return Err(Duckerror::Invalidargument(format!(
            "synthetic-writer: unknown table '{table}' (only '{TABLE_NAME}' is served)"
        )));
    }
    Ok(())
}

fn dv_to_text(v: &Duckvalue) -> Result<String, Duckerror> {
    match v {
        Duckvalue::Text(t) => Ok(t.clone()),
        Duckvalue::Null => Ok(String::new()),
        other => Err(Duckerror::Invalidargument(format!(
            "synthetic-writer: expected TEXT for kv column, got {other:?}"
        ))),
    }
}

fn current_view_mut(st: &mut KvState) -> &mut KvRows {
    if st.shadow.is_some() {
        st.shadow.as_mut().unwrap()
    } else {
        &mut st.committed
    }
}

struct Component;

impl Guest for Component {
    fn begin_transaction(_handle: u32, catalog: u32) -> Result<u32, Duckerror> {
        require_catalog(catalog)?;
        let mut st = state().lock().map_err(|_| poison())?;
        if st.active_txn.is_some() {
            return Err(Duckerror::Invalidstate(
                "synthetic-writer: nested transaction on same catalog".to_string(),
            ));
        }
        let txn = st.next_txn.wrapping_add(1).max(1);
        st.next_txn = txn;
        st.active_txn = Some(txn);
        st.shadow = Some(st.committed.clone());
        Ok(txn)
    }

    fn commit_transaction(_handle: u32, txn: u32) -> Result<(), Duckerror> {
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        if let Some(shadow) = st.shadow.take() {
            st.committed = shadow;
        }
        st.active_txn = None;
        Ok(())
    }

    fn rollback_transaction(_handle: u32, txn: u32) -> Result<(), Duckerror> {
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        st.shadow = None;
        st.active_txn = None;
        Ok(())
    }

    fn create_table(
        _handle: u32,
        txn: u32,
        table: String,
        _columns: Vec<Columndef>,
    ) -> Result<(), Duckerror> {
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        if table != TABLE_NAME {
            return Err(Duckerror::Invalidargument(format!(
                "synthetic-writer: unknown table '{table}' (only '{TABLE_NAME}' is served)"
            )));
        }
        // Schema is fixed to (key TEXT, value TEXT). The caller may pass any
        // `columns` list; we ignore its shape and only record that the table
        // has been created.
        st.created = true;
        Ok(())
    }

    fn insert_rows(
        _handle: u32,
        txn: u32,
        table: String,
        rows: Vec<Vec<Duckvalue>>,
    ) -> Result<u64, Duckerror> {
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        require_table(&st, &table)?;
        let mut inserted: u64 = 0;
        for row in &rows {
            if row.len() != 2 {
                return Err(Duckerror::Invalidargument(format!(
                    "synthetic-writer: kv_store INSERT expects 2 columns, got {}",
                    row.len()
                )));
            }
            let key = dv_to_text(&row[0])?;
            let value = dv_to_text(&row[1])?;
            let view = current_view_mut(&mut st);
            let rid = view.next_rowid.wrapping_add(1).max(1);
            view.next_rowid = rid;
            view.rows.insert(rid, (key, value));
            inserted += 1;
        }
        Ok(inserted)
    }

    fn delete_rows(
        _handle: u32,
        txn: u32,
        table: String,
        rowids: Vec<i64>,
    ) -> Result<u64, Duckerror> {
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        require_table(&st, &table)?;
        let mut deleted: u64 = 0;
        for rid in &rowids {
            if current_view_mut(&mut st).rows.remove(rid).is_some() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    fn update_rows(
        _handle: u32,
        txn: u32,
        table: String,
        rowids: Vec<i64>,
        rows: Vec<Vec<Duckvalue>>,
    ) -> Result<u64, Duckerror> {
        if rowids.len() != rows.len() {
            return Err(Duckerror::Invalidargument(format!(
                "synthetic-writer: UPDATE rowids ({}) / rows ({}) length mismatch",
                rowids.len(),
                rows.len()
            )));
        }
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        require_table(&st, &table)?;
        let mut updated: u64 = 0;
        for (rid, row) in rowids.iter().zip(rows.iter()) {
            if row.len() != 2 {
                return Err(Duckerror::Invalidargument(format!(
                    "synthetic-writer: kv_store UPDATE expects 2 columns, got {}",
                    row.len()
                )));
            }
            let key = dv_to_text(&row[0])?;
            let value = dv_to_text(&row[1])?;
            let view = current_view_mut(&mut st);
            if view.rows.insert(*rid, (key, value)).is_some() {
                updated += 1;
            }
        }
        Ok(updated)
    }
}

bindings::export!(Component with_types_in bindings);
