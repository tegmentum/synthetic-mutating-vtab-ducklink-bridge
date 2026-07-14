//! Synthetic ducklink storage bridge (test-only).
//!
//! FULL-STACK writer + reader bridge that lets the ducklink CLI drive
//! `ATTACH ':memory:' AS kv (TYPE synthetic_mutating);` end-to-end. Exports
//! four halves of the `duckdb:extension@4.0.0` contract, all backed by the
//! SAME in-guest `Mutex<KvState>`:
//!
//!   * `guest` -- `load()` calls `storage.register-storage("synthetic_mutating",
//!     1, None)` so the CLI's storage-backend lookup routes ATTACH TYPE
//!     `synthetic_mutating` here.
//!   * `callback-dispatch` -- stubs (this bridge has no scalar/table/pragma
//!     capabilities; it's a pure storage backend).
//!   * `storage-dispatch` -- read-side: attach / list-tables / table-columns /
//!     scan-open / scan-next / scan-close / detach + attach-blob no-op.
//!   * `storage-write-dispatch` -- write-side: begin/commit/rollback +
//!     create-table + insert/update/delete-rows.
//!
//! Semantics: a synthetic `kv_store(key TEXT, value TEXT)` backend. State is
//! a process-local `Mutex<KvState>` holding:
//!   * `committed`: the last-committed BTreeMap<rowid, (key, value)>.
//!   * `shadow`: the current transaction's mutable snapshot, promoted on
//!     commit / discarded on rollback.
//! No savepoints (DuckDB's TransactionManager doesn't use SAVEPOINT the same
//! way SQLite's vtab-update does).
//!
//! Read side observes the shadow (if a txn is open) so read-after-write
//! within a transaction reflects the pending mutations, exactly like the
//! sqlink sibling (`synthetic-mutating-vtab-bridge`).
//!
//! History: v0.1.0 was write-only (only storage-write-dispatch); v0.2.0
//! adds guest + storage-dispatch so the CLI can drive the whole ATTACH
//! flow, not just the write-boundary test. See `wit/world.wit`.
#![allow(unsafe_op_in_unsafe_fn)]

mod bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "writer",
        generate_all,
    });
}

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

use bindings::duckdb::extension::storage::{self, CompareOp, ScanFilter};
use bindings::duckdb::extension::types::{Capabilitykind, Columndef, Duckerror, Duckvalue, Loadresult, Logicaltype};
use bindings::exports::duckdb::extension::callback_dispatch::{self, Guest as CallbackGuest};
use bindings::exports::duckdb::extension::guest::Guest as GuestGuest;
use bindings::exports::duckdb::extension::storage_dispatch::{Guest as StorageDispatchGuest, ScanRequest};
use bindings::exports::duckdb::extension::storage_write_dispatch::Guest as StorageWriteDispatchGuest;

// -----------------------------------------------------------
// Backing store: rowid -> (key, value).
//
// A single global store keyed on rowid mirrors what a persistent
// storage backend would look like without a real disk. The txn
// state (`shadow`) captures the pre-txn snapshot so rollback can
// restore it; commit promotes shadow -> committed.
// -----------------------------------------------------------

/// The storage-extension callback handle handed back by every host call.
/// A single registered backend, so this is fixed.
const HANDLE: u32 = 1;
/// The one served catalog. `storage-attach` allocates it on first use;
/// `storage-detach` clears the flag.
const CATALOG_HANDLE: u32 = 1;
/// The one served table. All storage-write calls check against it.
const TABLE_NAME: &str = "kv_store";

#[derive(Default, Clone)]
struct KvRows {
    rows: BTreeMap<i64, (String, String)>,
    next_rowid: i64,
}

impl KvRows {
    fn ordered(&self) -> Vec<(i64, String, String)> {
        self.rows
            .iter()
            .map(|(rid, (k, v))| (*rid, k.clone(), v.clone()))
            .collect()
    }
}

#[derive(Default)]
struct KvState {
    committed: KvRows,
    /// Present between `begin-transaction` and `commit`/`rollback`. Holds the
    /// pre-txn snapshot so rollback can restore it and in-txn readers see
    /// the pending writes.
    shadow: Option<KvRows>,
    /// Monotonic txn counter. Handed out by `begin-transaction` and
    /// checked by every subsequent write call.
    next_txn: u32,
    /// The currently-open txn handle, if any. DuckDB's TransactionManager
    /// serializes writes per catalog, so at most one live txn at a time.
    active_txn: Option<u32>,
    /// True iff `create-table("kv_store", ...)` has been called at any
    /// point (across sessions of this process). Gates `list-tables` /
    /// `table-columns` visibility and the write-side table checks.
    /// The schema is fixed to (key TEXT, value TEXT).
    created: bool,
    /// True iff a `storage-attach` has been called and not yet
    /// `storage-detach`-ed. Multiple attaches under the same
    /// process share the one catalog record (CATALOG_HANDLE).
    attached: bool,
    /// Per-scan cursors keyed by scan-id. Each cursor is a snapshot at
    /// scan-open time (of the view visible then -- shadow if in-txn,
    /// committed otherwise) plus a projection map and a position.
    scans: HashMap<u32, Cursor>,
    next_scan: u32,
}

struct Cursor {
    /// Materialized rows in projection order.
    rows: Vec<Vec<Duckvalue>>,
    /// Next row index to emit.
    pos: usize,
}

fn state() -> &'static Mutex<KvState> {
    static S: OnceLock<Mutex<KvState>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(KvState::default()))
}

fn poison() -> Duckerror {
    Duckerror::Internal("synthetic-storage: state mutex poisoned".to_string())
}

fn require_handle(handle: u32) -> Result<(), Duckerror> {
    if handle == HANDLE {
        Ok(())
    } else {
        Err(Duckerror::Invalidargument(format!(
            "synthetic-storage: unknown callback handle {handle} (only {HANDLE} is served)"
        )))
    }
}

fn require_catalog(catalog: u32) -> Result<(), Duckerror> {
    if catalog == CATALOG_HANDLE {
        Ok(())
    } else {
        Err(Duckerror::Invalidargument(format!(
            "synthetic-storage: unknown catalog handle {catalog} (only {CATALOG_HANDLE} is served)"
        )))
    }
}

fn require_txn(st: &KvState, txn: u32) -> Result<(), Duckerror> {
    match st.active_txn {
        Some(t) if t == txn => Ok(()),
        Some(other) => Err(Duckerror::Invalidstate(format!(
            "synthetic-storage: txn {txn} not active (current active txn is {other})"
        ))),
        None => Err(Duckerror::Invalidstate(format!(
            "synthetic-storage: txn {txn} not active (no live transaction)"
        ))),
    }
}

fn require_table(st: &KvState, table: &str) -> Result<(), Duckerror> {
    if !st.created {
        return Err(Duckerror::Invalidstate(format!(
            "synthetic-storage: table '{table}' has not been created"
        )));
    }
    if table != TABLE_NAME {
        return Err(Duckerror::Invalidargument(format!(
            "synthetic-storage: unknown table '{table}' (only '{TABLE_NAME}' is served)"
        )));
    }
    Ok(())
}

fn dv_to_text(v: &Duckvalue) -> Result<String, Duckerror> {
    match v {
        Duckvalue::Text(t) => Ok(t.clone()),
        Duckvalue::Null => Ok(String::new()),
        other => Err(Duckerror::Invalidargument(format!(
            "synthetic-storage: expected TEXT for kv column, got {other:?}"
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

/// Visible view for a read at the current moment: shadow (uncommitted) if
/// a txn is open, otherwise committed. Mirrors the sqlink sibling.
fn visible_view(st: &KvState) -> &KvRows {
    st.shadow.as_ref().unwrap_or(&st.committed)
}

struct Component;

// -----------------------------------------------------------
// guest: load-time hook. Declares the storage backend so the
// ducklink CLI's storage-backend lookup can route ATTACH TYPE
// `synthetic_mutating` to this component.
// -----------------------------------------------------------

impl GuestGuest for Component {
    fn load() -> Result<Loadresult, Duckerror> {
        // Declare the ATTACH TYPE name. Also declares `catalog` as the
        // required capability so the host knows to keep this component
        // instantiated across the session (a scalar-only backend can be
        // spun up per-call; a storage backend must persist).
        storage::register_storage("synthetic_mutating", HANDLE, None)?;
        Ok(Loadresult {
            name: "synthetic_mutating".to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            requires: vec![Capabilitykind::Catalog],
        })
    }

    fn reconfigure(_keys: Vec<String>) -> Result<bool, Duckerror> {
        Ok(false)
    }

    fn shutdown() -> Result<bool, Duckerror> {
        Ok(false)
    }
}

// -----------------------------------------------------------
// callback-dispatch: this bridge exports no scalar / table /
// pragma / cast fns. Every arm is Unsupported.
// -----------------------------------------------------------

impl CallbackGuest for Component {
    fn call_scalar_batch_col(
        _handle: u32,
        _args: Vec<callback_dispatch::Colvec>,
        _ctx: bindings::duckdb::extension::types::Invokeinfo,
    ) -> Result<callback_dispatch::Colvec, Duckerror> {
        Err(Duckerror::Unsupported(
            "synthetic-storage: no scalar functions".to_string(),
        ))
    }

    fn call_aggregate_col(
        _handle: u32,
        _args: Vec<callback_dispatch::Colvec>,
    ) -> Result<Duckvalue, Duckerror> {
        Err(Duckerror::Unsupported(
            "synthetic-storage: no aggregates".to_string(),
        ))
    }

    fn call_cast_col(
        _handle: u32,
        _arg: callback_dispatch::Colvec,
    ) -> Result<callback_dispatch::Colvec, Duckerror> {
        Err(Duckerror::Unsupported(
            "synthetic-storage: no casts".to_string(),
        ))
    }

    fn call_scalar(
        _handle: u32,
        _args: Vec<Duckvalue>,
        _ctx: bindings::duckdb::extension::types::Invokeinfo,
    ) -> Result<Duckvalue, Duckerror> {
        Err(Duckerror::Unsupported(
            "synthetic-storage: no scalar functions".to_string(),
        ))
    }

    fn call_table(
        _handle: u32,
        _args: Vec<Duckvalue>,
    ) -> Result<Vec<Vec<Duckvalue>>, Duckerror> {
        Err(Duckerror::Unsupported(
            "synthetic-storage: no table functions".to_string(),
        ))
    }

    fn call_pragma(
        _handle: u32,
        _args: Vec<Duckvalue>,
    ) -> Result<Option<Duckvalue>, Duckerror> {
        Err(Duckerror::Unsupported(
            "synthetic-storage: no pragmas".to_string(),
        ))
    }

    fn call_cast(
        _handle: u32,
        _value: Duckvalue,
    ) -> Result<Duckvalue, Duckerror> {
        Err(Duckerror::Unsupported(
            "synthetic-storage: no casts".to_string(),
        ))
    }
}

// -----------------------------------------------------------
// storage-dispatch: read-side. Attach / list / column-schema /
// scan-open / scan-next / scan-close / detach. Read observes
// the shadow (if in-txn) so read-after-write within a
// transaction reflects the pending mutations.
// -----------------------------------------------------------

impl StorageDispatchGuest for Component {
    fn storage_attach(
        handle: u32,
        _dsn: String,
        _options: Vec<(String, String)>,
    ) -> Result<u32, Duckerror> {
        require_handle(handle)?;
        let mut st = state().lock().map_err(|_| poison())?;
        // Single-catalog model: repeat attaches under the same process
        // return the same fixed CATALOG_HANDLE. The synthetic kv_store
        // state is process-global regardless of the ATTACH dsn.
        st.attached = true;
        Ok(CATALOG_HANDLE)
    }

    fn attach_blob(
        handle: u32,
        _dsn: String,
        _bytes: Vec<u8>,
    ) -> Result<(), Duckerror> {
        require_handle(handle)?;
        // Synthetic kv has no on-disk representation; staged blobs are
        // ignored. Accepting the call keeps a caller that BLOB-stages
        // before ATTACH from erroring.
        Ok(())
    }

    fn storage_list_tables(
        handle: u32,
        catalog: u32,
    ) -> Result<Vec<String>, Duckerror> {
        require_handle(handle)?;
        require_catalog(catalog)?;
        let st = state().lock().map_err(|_| poison())?;
        if st.created {
            Ok(vec![TABLE_NAME.to_string()])
        } else {
            Ok(Vec::new())
        }
    }

    fn storage_table_columns(
        handle: u32,
        catalog: u32,
        table: String,
    ) -> Result<Vec<Columndef>, Duckerror> {
        require_handle(handle)?;
        require_catalog(catalog)?;
        let st = state().lock().map_err(|_| poison())?;
        // Pre-CREATE-TABLE, DuckDB calls this to check for a name clash. Returning
        // an error stashes a stale message in wasm-storage's last-error slot that
        // the C++ scan-fill EOF path later re-surfaces as a fabricated failure.
        // Return an empty column list instead: the C++ core's GetOrLoadTable
        // treats an empty column blob as "table not found" and moves on, and
        // last-error stays clean. Post-CREATE-TABLE calls hit the second branch.
        if !st.created || table != TABLE_NAME {
            return Ok(Vec::new());
        }
        Ok(kv_schema())
    }

    fn storage_scan_open(
        handle: u32,
        catalog: u32,
        request: ScanRequest,
    ) -> Result<u32, Duckerror> {
        require_handle(handle)?;
        require_catalog(catalog)?;
        let mut st = state().lock().map_err(|_| poison())?;
        require_table(&st, &request.table)?;

        // Projection: full column list is [key, value]. Empty projection
        // means all columns in natural order.
        let proj: Vec<u32> = if request.projection.is_empty() {
            vec![0, 1]
        } else {
            request.projection.clone()
        };
        for &i in &proj {
            if i >= 2 {
                return Err(Duckerror::Invalidargument(format!(
                    "synthetic-storage: projection index {i} out of range (schema has 2 cols)"
                )));
            }
        }

        // Snapshot the visible view now. Later mutations don't affect an
        // already-open scan, matching sqlitewasm-component's semantic
        // (which materializes the full resultset at scan-open time).
        //
        // Filters MUST be applied here: DuckDB's WasmTableEntry publishes
        // `filter_pushdown = true`, so the engine does NOT re-apply the
        // predicates it pushes down -- an unfiltered result set is treated
        // as authoritative and the WHERE clause never fires downstream.
        // Apply predicates AND-wise; a filter the bridge can't evaluate
        // (unknown column, non-text RHS) returns true so the row survives
        // to a re-application later in the query plan.
        let src = visible_view(&st).ordered();
        let limit = request.limit.map(|n| n as usize).unwrap_or(usize::MAX);
        let mut rows: Vec<Vec<Duckvalue>> = Vec::new();
        for (_rid, k, v) in src.iter() {
            if !request.filters.iter().all(|f| eval_filter(k, v, f)) {
                continue;
            }
            let cells: Vec<Duckvalue> = proj
                .iter()
                .map(|&i| match i {
                    0 => Duckvalue::Text(k.clone()),
                    1 => Duckvalue::Text(v.clone()),
                    _ => unreachable!(),
                })
                .collect();
            rows.push(cells);
            if rows.len() >= limit {
                break;
            }
        }

        let scan_id = st.next_scan.wrapping_add(1).max(1);
        st.next_scan = scan_id;
        st.scans.insert(scan_id, Cursor { rows, pos: 0 });
        Ok(scan_id)
    }

    fn storage_scan_next(
        handle: u32,
        scan: u32,
        max_rows: u32,
    ) -> Result<Vec<Vec<Duckvalue>>, Duckerror> {
        require_handle(handle)?;
        let mut st = state().lock().map_err(|_| poison())?;
        let cur = st.scans.get_mut(&scan).ok_or_else(|| {
            Duckerror::Invalidstate(format!("synthetic-storage: unknown scan handle {scan}"))
        })?;
        let end = (cur.pos + max_rows as usize).min(cur.rows.len());
        let batch: Vec<Vec<Duckvalue>> = cur.rows[cur.pos..end].to_vec();
        cur.pos = end;
        Ok(batch)
    }

    fn storage_scan_close(handle: u32, scan: u32) -> Result<bool, Duckerror> {
        require_handle(handle)?;
        let mut st = state().lock().map_err(|_| poison())?;
        Ok(st.scans.remove(&scan).is_some())
    }

    fn storage_detach(handle: u32, catalog: u32) -> Result<bool, Duckerror> {
        require_handle(handle)?;
        require_catalog(catalog)?;
        let mut st = state().lock().map_err(|_| poison())?;
        let was = st.attached;
        st.attached = false;
        // Purposely do NOT clear `committed` / `created` -- they're the
        // synthetic kv's "on-disk" state; a re-ATTACH under the same
        // process re-exposes the prior rows, matching how a real
        // storage backend behaves.
        Ok(was)
    }
}

fn kv_schema() -> Vec<Columndef> {
    vec![
        Columndef {
            name: "key".to_string(),
            logical: Logicaltype::Text,
        },
        Columndef {
            name: "value".to_string(),
            logical: Logicaltype::Text,
        },
    ]
}

/// Column-scoped access for filter evaluation. Column 0 is key, 1 is value;
/// anything else is an invariant break in the caller.
fn kv_cell<'a>(k: &'a str, v: &'a str, column: u32) -> Option<&'a str> {
    match column {
        0 => Some(k),
        1 => Some(v),
        _ => None,
    }
}

/// Evaluate one pushed-down filter against a materialized (key, value) pair.
/// Text-only comparisons; a non-text `duckvalue` on the RHS returns `true`
/// (the engine will re-apply and reject). is-null / is-not-null: our columns
/// are declared NOT NULL semantically (INSERT with NULL is stored as ""),
/// so is-null is always false and is-not-null always true.
fn eval_filter(k: &str, v: &str, filter: &ScanFilter) -> bool {
    let cell = match kv_cell(k, v, filter.column) {
        Some(c) => c,
        None => return true, // unknown column -> pass, engine re-applies
    };
    match filter.op {
        CompareOp::IsNull => false,
        CompareOp::IsNotNull => true,
        op => {
            let rhs = match &filter.value {
                Duckvalue::Text(t) => t.as_str(),
                Duckvalue::Null => return matches!(op, CompareOp::Ne), // != NULL is TRUE-ish
                _ => return true, // non-text RHS: let the engine re-apply
            };
            match op {
                CompareOp::Eq => cell == rhs,
                CompareOp::Ne => cell != rhs,
                CompareOp::Lt => cell < rhs,
                CompareOp::Le => cell <= rhs,
                CompareOp::Gt => cell > rhs,
                CompareOp::Ge => cell >= rhs,
                CompareOp::IsNull | CompareOp::IsNotNull => unreachable!(),
            }
        }
    }
}

// -----------------------------------------------------------
// storage-write-dispatch: write-side. Transactions + CREATE
// TABLE + INSERT / UPDATE / DELETE. Uses the shadow-KvRows
// pattern: begin snapshots committed into shadow; commit
// promotes shadow into committed; rollback drops shadow.
// -----------------------------------------------------------

impl StorageWriteDispatchGuest for Component {
    fn begin_transaction(handle: u32, catalog: u32) -> Result<u32, Duckerror> {
        require_handle(handle)?;
        require_catalog(catalog)?;
        let mut st = state().lock().map_err(|_| poison())?;
        if st.active_txn.is_some() {
            return Err(Duckerror::Invalidstate(
                "synthetic-storage: nested transaction on same catalog".to_string(),
            ));
        }
        let txn = st.next_txn.wrapping_add(1).max(1);
        st.next_txn = txn;
        st.active_txn = Some(txn);
        st.shadow = Some(st.committed.clone());
        Ok(txn)
    }

    fn commit_transaction(handle: u32, txn: u32) -> Result<(), Duckerror> {
        require_handle(handle)?;
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        if let Some(shadow) = st.shadow.take() {
            st.committed = shadow;
        }
        st.active_txn = None;
        Ok(())
    }

    fn rollback_transaction(handle: u32, txn: u32) -> Result<(), Duckerror> {
        require_handle(handle)?;
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        st.shadow = None;
        st.active_txn = None;
        Ok(())
    }

    fn create_table(
        handle: u32,
        txn: u32,
        table: String,
        _columns: Vec<Columndef>,
    ) -> Result<(), Duckerror> {
        require_handle(handle)?;
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        if table != TABLE_NAME {
            return Err(Duckerror::Invalidargument(format!(
                "synthetic-storage: unknown table '{table}' (only '{TABLE_NAME}' is served)"
            )));
        }
        // Schema is fixed to (key TEXT, value TEXT). The caller may pass any
        // `columns` list; we ignore its shape and only record that the table
        // has been created.
        st.created = true;
        Ok(())
    }

    fn insert_rows(
        handle: u32,
        txn: u32,
        table: String,
        rows: Vec<Vec<Duckvalue>>,
    ) -> Result<u64, Duckerror> {
        require_handle(handle)?;
        let mut st = state().lock().map_err(|_| poison())?;
        require_txn(&st, txn)?;
        require_table(&st, &table)?;
        let mut inserted: u64 = 0;
        for row in &rows {
            if row.len() != 2 {
                return Err(Duckerror::Invalidargument(format!(
                    "synthetic-storage: kv_store INSERT expects 2 columns, got {}",
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
        handle: u32,
        txn: u32,
        table: String,
        rowids: Vec<i64>,
    ) -> Result<u64, Duckerror> {
        require_handle(handle)?;
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
        handle: u32,
        txn: u32,
        table: String,
        rowids: Vec<i64>,
        rows: Vec<Vec<Duckvalue>>,
    ) -> Result<u64, Duckerror> {
        require_handle(handle)?;
        if rowids.len() != rows.len() {
            return Err(Duckerror::Invalidargument(format!(
                "synthetic-storage: UPDATE rowids ({}) / rows ({}) length mismatch",
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
                    "synthetic-storage: kv_store UPDATE expects 2 columns, got {}",
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
