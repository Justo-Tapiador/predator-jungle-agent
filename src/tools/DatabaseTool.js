/**
 * DatabaseTool.js
 * Database operations for the PREDATOR agent system.
 * Supports SQLite via better-sqlite3 when available, with a full
 * in-memory store as fallback. All methods are async and return
 * structured results.
 */

const DEFAULT_TIMEOUT = 10000; // 10s query timeout

class DatabaseTool {
  constructor(options = {}) {
    this.id = 'database';
    this.name = 'DatabaseTool';
    this.description = 'Execute database queries (SQLite or in-memory fallback)';
    this.defaultTimeout = options.defaultTimeout || DEFAULT_TIMEOUT;

    // Connection state
    this._db = null;
    this._driver = null; // 'better-sqlite3' | 'in-memory'
    this._connectionString = null;
    this._connected = false;

    // In-memory fallback store
    this._tables = new Map(); // tableName -> { columns: [], rows: [] }
    this._autoIncrement = new Map(); // tableName -> nextId

    // Lazy-load better-sqlite3 on first connect
    this._sqliteAvailable = false;
    this._BetterSqlite3 = null;
    this._sqliteLoaded = false;
  }

  async _loadSQLite() {
    if (this._sqliteLoaded) return;
    this._sqliteLoaded = true;
    try {
      // eslint-disable-next-line import/no-unresolved
      this._BetterSqlite3 = await import('better-sqlite3');
      this._sqliteAvailable = true;
    } catch {
      // better-sqlite3 not available — will use in-memory fallback
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  _result(success, result, error, startTime) {
    return {
      success,
      ...(result !== undefined && { result }),
      ...(error && { error }),
      duration: Date.now() - startTime,
    };
  }

  _requireConnection() {
    if (!this._connected) {
      throw new Error('Not connected to a database. Call connect() first.');
    }
  }

  // ── In-Memory SQL Parser (subset) ────────────────────────────────────

  _parseCreateTable(sql) {
    const match = sql.match(
      /CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["']?(\w+)["']?\s*\(([\s\S]*)\)/i,
    );
    if (!match) throw new Error('Invalid CREATE TABLE syntax');

    const tableName = match[1];
    const columnsPart = match[2];
    const columns = columnsPart
      .split(',')
      .map((col) => col.trim().split(/\s+/)[0].replace(/["']/g, ''))
      .filter(Boolean);

    return { tableName, columns };
  }

  _parseInsert(sql) {
    const match = sql.match(
      /INSERT\s+INTO\s+["']?(\w+)["']?\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)/i,
    );
    if (!match) throw new Error('Invalid INSERT syntax');

    const tableName = match[1];
    const columns = match[2].split(',').map((c) => c.trim().replace(/["']/g, ''));
    const values = match[3].split(',').map((v) => {
      v = v.trim();
      if (v === 'NULL') return null;
      if (v === 'TRUE') return true;
      if (v === 'FALSE') return false;
      if (/^["']/.test(v)) return v.slice(1, -1);
      if (!isNaN(Number(v))) return Number(v);
      return v;
    });

    return { tableName, columns, values };
  }

  _parseSelect(sql) {
    const match = sql.match(
      /SELECT\s+([\s\S]+?)\s+FROM\s+["']?(\w+)["']?(?:\s+WHERE\s+([\s\S]+?))?(?:\s+ORDER\s+BY\s+([\s\S]+?))?(?:\s+LIMIT\s+(\d+))?\s*$/i,
    );
    if (!match) throw new Error('Invalid SELECT syntax');

    const columnsRaw = match[1].trim();
    const tableName = match[2];
    const whereClause = match[3] || null;
    const orderByClause = match[4] || null;
    const limit = match[5] ? parseInt(match[5], 10) : null;

    let columns;
    if (columnsRaw === '*') {
      const table = this._tables.get(tableName);
      columns = table ? table.columns : [];
    } else {
      columns = columnsRaw.split(',').map((c) => c.trim().replace(/["']/g, ''));
    }

    return { tableName, columns, whereClause, orderByClause, limit };
  }

  _parseUpdate(sql) {
    const match = sql.match(
      /UPDATE\s+["']?(\w+)["']?\s+SET\s+([\s\S]+?)\s+WHERE\s+([\s\S]+)\s*$/i,
    );
    if (!match) throw new Error('Invalid UPDATE syntax (WHERE clause required)');

    const tableName = match[1];
    const setParts = match[2].split(',').map((part) => {
      const [col, val] = part.split('=').map((s) => s.trim());
      let parsedVal = val.replace(/["']/g, '');
      if (parsedVal === 'NULL') parsedVal = null;
      else if (parsedVal === 'TRUE') parsedVal = true;
      else if (parsedVal === 'FALSE') parsedVal = false;
      else if (!isNaN(Number(parsedVal))) parsedVal = Number(parsedVal);
      return { column: col, value: parsedVal };
    });
    const whereClause = match[3];

    return { tableName, setParts, whereClause };
  }

  _parseDelete(sql) {
    const match = sql.match(/DELETE\s+FROM\s+["']?(\w+)["']?\s+WHERE\s+([\s\S]+)\s*$/i);
    if (!match) throw new Error('Invalid DELETE syntax (WHERE clause required)');

    return { tableName: match[1], whereClause: match[2] };
  }

  _evaluateWhere(row, whereClause) {
    if (!whereClause) return true;

    // Simple condition evaluator: column = value, column > value, etc.
    const operators = ['>=', '<=', '!=', '<>', '=', '>', '<', 'LIKE', 'IS NOT', 'IS'];
    const upperWhere = whereClause.toUpperCase();

    for (const op of operators) {
      const idx = upperWhere.indexOf(op);
      if (idx === -1) continue;

      const column = whereClause.substring(0, idx).trim().replace(/["']/g, '');
      let valueStr = whereClause.substring(idx + op.length).trim();

      // Parse value
      let compareValue;
      if (valueStr.toUpperCase() === 'NULL') compareValue = null;
      else if (valueStr.toUpperCase() === 'TRUE') compareValue = true;
      else if (valueStr.toUpperCase() === 'FALSE') compareValue = false;
      else {
        compareValue = valueStr.replace(/["']/g, '');
        if (!isNaN(Number(compareValue))) compareValue = Number(compareValue);
      }

      const rowValue = row[column];

      switch (op) {
        case '=':
          return rowValue == compareValue;
        case '!=':
        case '<>':
          return rowValue != compareValue;
        case '>':
          return rowValue > compareValue;
        case '<':
          return rowValue < compareValue;
        case '>=':
          return rowValue >= compareValue;
        case '<=':
          return rowValue <= compareValue;
        case 'IS':
          return rowValue === compareValue;
        case 'IS NOT':
          return rowValue !== compareValue;
        case 'LIKE': {
          const pattern = String(compareValue).replace(/%/g, '.*').replace(/_/g, '.');
          return new RegExp(`^${pattern}$`, 'i').test(String(rowValue));
        }
      }
    }

    // Fallback: try AND conditions
    if (upperWhere.includes(' AND ')) {
      const parts = whereClause.split(/\s+AND\s+/i);
      return parts.every((part) => this._evaluateWhere(row, part.trim()));
    }

    return true;
  }

  // ── In-Memory Execution ──────────────────────────────────────────────

  _executeInMemory(sql, params = []) {
    const trimmed = sql.trim();

    if (/^\s*CREATE\s+TABLE/i.test(trimmed)) {
      const { tableName, columns } = this._parseCreateTable(trimmed);
      if (this._tables.has(tableName)) {
        throw new Error(`Table "${tableName}" already exists`);
      }
      this._tables.set(tableName, { columns, rows: [] });
      this._autoIncrement.set(tableName, 1);
      return { changes: 0, tableCreated: tableName };
    }

    if (/^\s*INSERT/i.test(trimmed)) {
      const { tableName, columns, values } = this._parseInsert(trimmed);
      const table = this._tables.get(tableName);
      if (!table) throw new Error(`Table "${tableName}" does not exist`);

      const row = {};
      columns.forEach((col, i) => {
        row[col] = values[i] !== undefined ? values[i] : null;
      });

      // Auto-increment for 'id' column
      if (columns.includes('id') && row.id === null) {
        row.id = this._autoIncrement.get(tableName);
        this._autoIncrement.set(tableName, this._autoIncrement.get(tableName) + 1);
      }

      table.rows.push(row);
      return { changes: 1, lastInsertRowid: row.id || table.rows.length };
    }

    if (/^\s*SELECT/i.test(trimmed)) {
      const { tableName, columns, whereClause, orderByClause, limit } = this._parseSelect(trimmed);
      const table = this._tables.get(tableName);
      if (!table) throw new Error(`Table "${tableName}" does not exist`);

      let rows = table.rows.filter((row) => this._evaluateWhere(row, whereClause));

      // Order by
      if (orderByClause) {
        const orderCol = orderByClause.trim().split(/\s+/)[0].replace(/["']/g, '');
        const orderDir = orderByClause.toUpperCase().includes('DESC') ? -1 : 1;
        rows.sort((a, b) => {
          if (a[orderCol] < b[orderCol]) return -1 * orderDir;
          if (a[orderCol] > b[orderCol]) return 1 * orderDir;
          return 0;
        });
      }

      // Limit
      if (limit !== null) rows = rows.slice(0, limit);

      // Select columns
      const result = rows.map((row) => {
        const out = {};
        for (const col of columns) {
          out[col] = row[col] !== undefined ? row[col] : null;
        }
        return out;
      });

      return result;
    }

    if (/^\s*UPDATE/i.test(trimmed)) {
      const { tableName, setParts, whereClause } = this._parseUpdate(trimmed);
      const table = this._tables.get(tableName);
      if (!table) throw new Error(`Table "${tableName}" does not exist`);

      let changes = 0;
      for (const row of table.rows) {
        if (this._evaluateWhere(row, whereClause)) {
          for (const { column, value } of setParts) {
            row[column] = value;
          }
          changes++;
        }
      }
      return { changes };
    }

    if (/^\s*DELETE/i.test(trimmed)) {
      const { tableName, whereClause } = this._parseDelete(trimmed);
      const table = this._tables.get(tableName);
      if (!table) throw new Error(`Table "${tableName}" does not exist`);

      const before = table.rows.length;
      table.rows = table.rows.filter((row) => !this._evaluateWhere(row, whereClause));
      const changes = before - table.rows.length;
      return { changes };
    }

    throw new Error(`Unsupported SQL statement: ${trimmed.substring(0, 50)}...`);
  }

  // ── Public API ───────────────────────────────────────────────────────

  async connect(connectionString = ':memory:') {
    const start = Date.now();
    try {
      await this._loadSQLite();

      if (this._connected) {
        return this._result(false, undefined, 'Already connected. Call disconnect() first.', start);
      }

      this._connectionString = connectionString;

      if (connectionString === ':memory:' || !this._sqliteAvailable) {
        // Use in-memory fallback
        this._driver = 'in-memory';
        this._tables.clear();
        this._autoIncrement.clear();
        this._connected = true;
        return this._result(
          true,
          {
            driver: this._driver,
            connectionString,
            message: this._sqliteAvailable
              ? 'Connected to in-memory database'
              : 'Connected to in-memory database (better-sqlite3 not available)',
          },
          undefined,
          start,
        );
      }

      // Use better-sqlite3
      const db = new this._BetterSqlite3.default(connectionString);
      db.pragma('journal_mode = WAL');
      db.pragma('foreign_keys = ON');
      this._db = db;
      this._driver = 'better-sqlite3';
      this._connected = true;

      return this._result(
        true,
        {
          driver: this._driver,
          connectionString,
          message: 'Connected to SQLite database',
        },
        undefined,
        start,
      );
    } catch (err) {
      return this._result(false, undefined, `Connection failed: ${err.message}`, start);
    }
  }

  async query(sql, params = []) {
    const start = Date.now();
    try {
      this._requireConnection();

      if (!sql || typeof sql !== 'string') {
        return this._result(false, undefined, 'SQL query is required', start);
      }

      if (this._driver === 'in-memory') {
        const result = this._executeInMemory(sql, params);
        return this._result(
          true,
          {
            rows: Array.isArray(result) ? result : [],
            meta: !Array.isArray(result) ? result : { rowCount: result.length },
          },
          undefined,
          start,
        );
      }

      // better-sqlite3 execution
      const stmt = this._db.prepare(sql);
      const isSelect = /^\s*SELECT/i.test(sql);

      if (isSelect) {
        const rows = params.length > 0 ? stmt.all(...params) : stmt.all();
        return this._result(
          true,
          {
            rows,
            meta: { rowCount: rows.length },
          },
          undefined,
          start,
        );
      }

      const info = params.length > 0 ? stmt.run(...params) : stmt.run();
      return this._result(
        true,
        {
          rows: [],
          meta: {
            changes: info.changes,
            lastInsertRowid: info.lastInsertRowid,
          },
        },
        undefined,
        start,
      );
    } catch (err) {
      return this._result(false, undefined, `Query failed: ${err.message}`, start);
    }
  }

  async execute(sql, params = []) {
    const start = Date.now();
    try {
      this._requireConnection();

      if (!sql || typeof sql !== 'string') {
        return this._result(false, undefined, 'SQL statement is required', start);
      }

      if (this._driver === 'in-memory') {
        const result = this._executeInMemory(sql, params);
        return this._result(
          true,
          {
            changes: result.changes || 0,
            lastInsertRowid: result.lastInsertRowid || null,
          },
          undefined,
          start,
        );
      }

      // better-sqlite3 execution
      const stmt = this._db.prepare(sql);
      const info = params.length > 0 ? stmt.run(...params) : stmt.run();

      return this._result(
        true,
        {
          changes: info.changes,
          lastInsertRowid: info.lastInsertRowid,
        },
        undefined,
        start,
      );
    } catch (err) {
      return this._result(false, undefined, `Execute failed: ${err.message}`, start);
    }
  }

  async disconnect() {
    const start = Date.now();
    try {
      if (!this._connected) {
        return this._result(false, undefined, 'Not connected', start);
      }

      if (this._driver === 'better-sqlite3' && this._db) {
        this._db.close();
        this._db = null;
      }

      if (this._driver === 'in-memory') {
        this._tables.clear();
        this._autoIncrement.clear();
      }

      this._driver = null;
      this._connectionString = null;
      this._connected = false;

      return this._result(true, { disconnected: true }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `Disconnect failed: ${err.message}`, start);
    }
  }
}

export default DatabaseTool;
