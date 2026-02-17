import Database from "better-sqlite3";
import { randomUUID } from "crypto";
import path from "path";

const DB_PATH = path.join(process.cwd(), "data", "atobot.db");

let db: Database.Database;

export function initDatabase(): void {
  db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");

  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      display_name TEXT NOT NULL,
      role TEXT DEFAULT 'user' CHECK(role IN ('admin', 'user')),
      status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected', 'suspended')),
      alpaca_key_encrypted TEXT,
      alpaca_secret_encrypted TEXT,
      is_paper INTEGER DEFAULT 1,
      created_at TEXT DEFAULT (datetime('now')),
      approved_at TEXT,
      approved_by TEXT
    );

    CREATE TABLE IF NOT EXISTS trade_log (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      symbol TEXT NOT NULL,
      side TEXT NOT NULL,
      quantity REAL NOT NULL,
      price REAL,
      total_value REAL,
      strategy TEXT,
      status TEXT DEFAULT 'pending',
      alpaca_order_id TEXT,
      created_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (user_id) REFERENCES users(id)
    );
  `);

  console.log("[Database] SQLite initialized at", DB_PATH);
}

export interface DbUser {
  id: string;
  email: string;
  password_hash: string;
  display_name: string;
  role: "admin" | "user";
  status: "pending" | "approved" | "rejected" | "suspended";
  alpaca_key_encrypted: string | null;
  alpaca_secret_encrypted: string | null;
  is_paper: number;
  created_at: string;
  approved_at: string | null;
  approved_by: string | null;
}

export function getUserCount(): number {
  const row = db.prepare("SELECT COUNT(*) as count FROM users").get() as { count: number };
  return row.count;
}

export function createUser(email: string, passwordHash: string, displayName: string): DbUser {
  const id = randomUUID();
  const isFirstUser = getUserCount() === 0;
  const role = isFirstUser ? "admin" : "user";
  const status = isFirstUser ? "approved" : "pending";

  db.prepare(`
    INSERT INTO users (id, email, password_hash, display_name, role, status)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(id, email, passwordHash, displayName, role, status);

  return getUserById(id)!;
}

export function getUserByEmail(email: string): DbUser | null {
  return (db.prepare("SELECT * FROM users WHERE email = ?").get(email) as DbUser) || null;
}

export function getUserById(id: string): DbUser | null {
  return (db.prepare("SELECT * FROM users WHERE id = ?").get(id) as DbUser) || null;
}

export function getAllUsers(): Omit<DbUser, "password_hash" | "alpaca_key_encrypted" | "alpaca_secret_encrypted">[] {
  return db.prepare(
    "SELECT id, email, display_name, role, status, is_paper, created_at, approved_at, approved_by FROM users ORDER BY created_at DESC"
  ).all() as any[];
}

export function updateUserStatus(id: string, status: "approved" | "rejected" | "suspended", approvedBy?: string): void {
  if (status === "approved") {
    db.prepare("UPDATE users SET status = ?, approved_at = datetime('now'), approved_by = ? WHERE id = ?")
      .run(status, approvedBy || null, id);
  } else {
    db.prepare("UPDATE users SET status = ? WHERE id = ?").run(status, id);
  }
}

export function setUserAlpacaKeys(id: string, encryptedKey: string, encryptedSecret: string, isPaper: boolean): void {
  db.prepare("UPDATE users SET alpaca_key_encrypted = ?, alpaca_secret_encrypted = ?, is_paper = ? WHERE id = ?")
    .run(encryptedKey, encryptedSecret, isPaper ? 1 : 0, id);
}

export function getUserAlpacaKeys(id: string): { encryptedKey: string; encryptedSecret: string; isPaper: boolean } | null {
  const row = db.prepare("SELECT alpaca_key_encrypted, alpaca_secret_encrypted, is_paper FROM users WHERE id = ?").get(id) as any;
  if (!row || !row.alpaca_key_encrypted || !row.alpaca_secret_encrypted) return null;
  return {
    encryptedKey: row.alpaca_key_encrypted,
    encryptedSecret: row.alpaca_secret_encrypted,
    isPaper: !!row.is_paper,
  };
}

export function removeUserAlpacaKeys(id: string): void {
  db.prepare("UPDATE users SET alpaca_key_encrypted = NULL, alpaca_secret_encrypted = NULL WHERE id = ?").run(id);
}

export function getApprovedUsersWithKeys(): DbUser[] {
  return db.prepare(
    "SELECT * FROM users WHERE status = 'approved' AND alpaca_key_encrypted IS NOT NULL AND alpaca_secret_encrypted IS NOT NULL"
  ).all() as DbUser[];
}

export interface DbTrade {
  id: string;
  user_id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number | null;
  total_value: number | null;
  strategy: string | null;
  status: string;
  alpaca_order_id: string | null;
  created_at: string;
}

export function logTrade(
  userId: string, symbol: string, side: string, quantity: number,
  price: number | null, totalValue: number | null, strategy: string | null,
  status: string, alpacaOrderId: string | null
): void {
  const id = randomUUID();
  db.prepare(`
    INSERT INTO trade_log (id, user_id, symbol, side, quantity, price, total_value, strategy, status, alpaca_order_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(id, userId, symbol, side, quantity, price, totalValue, strategy, status, alpacaOrderId);
}

export function getUserTrades(userId: string, limit = 50): DbTrade[] {
  return db.prepare("SELECT * FROM trade_log WHERE user_id = ? ORDER BY created_at DESC LIMIT ?")
    .all(userId, limit) as DbTrade[];
}
