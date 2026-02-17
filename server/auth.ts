import type { Request, Response, NextFunction, Express } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import * as db from "./database";
import * as accountManager from "./accountManager";

// Extend Express Request to include user
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
        status: string;
        displayName: string;
      };
    }
  }
}

function getJwtSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret) throw new Error("JWT_SECRET environment variable not set");
  return secret;
}

function generateToken(user: db.DbUser): string {
  return jwt.sign(
    { id: user.id, email: user.email, role: user.role, status: user.status },
    getJwtSecret(),
    { expiresIn: "24h" }
  );
}

// --- Middleware ---

export function requireAuth(req: Request, res: Response, next: NextFunction): void {
  const header = req.headers.authorization;
  if (!header || !header.startsWith("Bearer ")) {
    res.status(401).json({ error: "Authentication required" });
    return;
  }

  const token = header.slice(7);
  try {
    const payload = jwt.verify(token, getJwtSecret()) as { id: string; email: string; role: string; status: string };
    const user = db.getUserById(payload.id);
    if (!user) {
      res.status(401).json({ error: "User not found" });
      return;
    }
    req.user = {
      id: user.id,
      email: user.email,
      role: user.role,
      status: user.status,
      displayName: user.display_name,
    };
    next();
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
}

export function requireApproved(req: Request, res: Response, next: NextFunction): void {
  if (!req.user) {
    res.status(401).json({ error: "Authentication required" });
    return;
  }
  if (req.user.status !== "approved") {
    res.status(403).json({ error: "Account pending approval" });
    return;
  }
  next();
}

export function requireAdmin(req: Request, res: Response, next: NextFunction): void {
  if (!req.user) {
    res.status(401).json({ error: "Authentication required" });
    return;
  }
  if (req.user.role !== "admin") {
    res.status(403).json({ error: "Admin access required" });
    return;
  }
  next();
}

// --- Route Registration ---

export function registerAuthRoutes(app: Express): void {
  // Register
  app.post("/api/auth/register", async (req: Request, res: Response) => {
    try {
      const { email, password, displayName } = req.body;
      if (!email || !password || !displayName) {
        return res.status(400).json({ error: "Email, password, and display name required" });
      }
      if (password.length < 6) {
        return res.status(400).json({ error: "Password must be at least 6 characters" });
      }

      const existing = db.getUserByEmail(email);
      if (existing) {
        return res.status(409).json({ error: "Email already registered" });
      }

      const passwordHash = await bcrypt.hash(password, 10);
      const user = db.createUser(email, passwordHash, displayName);
      const token = generateToken(user);

      console.log(`[Auth] New user registered: ${email} (role: ${user.role}, status: ${user.status})`);

      res.json({
        token,
        user: {
          id: user.id,
          email: user.email,
          displayName: user.display_name,
          role: user.role,
          status: user.status,
        },
      });
    } catch (err: any) {
      console.error("[Auth] Register error:", err.message);
      res.status(500).json({ error: "Registration failed" });
    }
  });

  // Login
  app.post("/api/auth/login", async (req: Request, res: Response) => {
    try {
      const { email, password } = req.body;
      if (!email || !password) {
        return res.status(400).json({ error: "Email and password required" });
      }

      const user = db.getUserByEmail(email);
      if (!user) {
        return res.status(401).json({ error: "Invalid email or password" });
      }

      const valid = await bcrypt.compare(password, user.password_hash);
      if (!valid) {
        return res.status(401).json({ error: "Invalid email or password" });
      }

      const token = generateToken(user);
      console.log(`[Auth] User logged in: ${email}`);

      res.json({
        token,
        user: {
          id: user.id,
          email: user.email,
          displayName: user.display_name,
          role: user.role,
          status: user.status,
        },
      });
    } catch (err: any) {
      console.error("[Auth] Login error:", err.message);
      res.status(500).json({ error: "Login failed" });
    }
  });

  // Get current user
  app.get("/api/auth/me", requireAuth, (req: Request, res: Response) => {
    res.json({
      id: req.user!.id,
      email: req.user!.email,
      displayName: req.user!.displayName,
      role: req.user!.role,
      status: req.user!.status,
    });
  });

  // Save API keys
  app.post("/api/auth/api-keys", requireAuth, requireApproved, async (req: Request, res: Response) => {
    try {
      const { alpacaKey, alpacaSecret, isPaper } = req.body;
      if (!alpacaKey || !alpacaSecret) {
        return res.status(400).json({ error: "API key and secret required" });
      }

      const encryptedKey = accountManager.encrypt(alpacaKey);
      const encryptedSecret = accountManager.encrypt(alpacaSecret);
      db.setUserAlpacaKeys(req.user!.id, encryptedKey, encryptedSecret, isPaper !== false);

      console.log(`[Auth] API keys saved for user ${req.user!.email}`);
      res.json({ success: true });
    } catch (err: any) {
      console.error("[Auth] Save API keys error:", err.message);
      res.status(500).json({ error: "Failed to save API keys" });
    }
  });

  // Get API key status
  app.get("/api/auth/api-keys/status", requireAuth, (req: Request, res: Response) => {
    const keys = db.getUserAlpacaKeys(req.user!.id);
    res.json({
      hasKeys: !!keys,
      isPaper: keys ? keys.isPaper : true,
    });
  });

  // Delete API keys
  app.delete("/api/auth/api-keys", requireAuth, (req: Request, res: Response) => {
    db.removeUserAlpacaKeys(req.user!.id);
    console.log(`[Auth] API keys removed for user ${req.user!.email}`);
    res.json({ success: true });
  });

  // Test API keys
  app.post("/api/auth/test-keys", requireAuth, async (req: Request, res: Response) => {
    try {
      const { alpacaKey, alpacaSecret, isPaper } = req.body;
      if (!alpacaKey || !alpacaSecret) {
        return res.status(400).json({ error: "API key and secret required" });
      }

      // Use the provided keys to test Alpaca connection
      const baseUrl = isPaper !== false
        ? "https://paper-api.alpaca.markets"
        : "https://api.alpaca.markets";

      const response = await fetch(`${baseUrl}/v2/account`, {
        headers: {
          "APCA-API-KEY-ID": alpacaKey,
          "APCA-API-SECRET-KEY": alpacaSecret,
        },
      });

      if (response.ok) {
        const account = await response.json() as any;
        res.json({
          success: true,
          equity: account.equity,
          status: account.status,
        });
      } else {
        const err = await response.text();
        res.json({ success: false, error: `Alpaca returned ${response.status}: ${err}` });
      }
    } catch (err: any) {
      res.json({ success: false, error: err.message });
    }
  });

  // --- Admin Routes ---

  app.get("/api/admin/users", requireAuth, requireAdmin, (req: Request, res: Response) => {
    const users = db.getAllUsers();
    res.json(users);
  });

  app.post("/api/admin/users/:id/approve", requireAuth, requireAdmin, (req: Request, res: Response) => {
    const user = db.getUserById(req.params.id);
    if (!user) return res.status(404).json({ error: "User not found" });
    db.updateUserStatus(req.params.id, "approved", req.user!.id);
    console.log(`[Auth] User ${user.email} approved by ${req.user!.email}`);
    res.json({ success: true });
  });

  app.post("/api/admin/users/:id/reject", requireAuth, requireAdmin, (req: Request, res: Response) => {
    const user = db.getUserById(req.params.id);
    if (!user) return res.status(404).json({ error: "User not found" });
    db.updateUserStatus(req.params.id, "rejected");
    console.log(`[Auth] User ${user.email} rejected by ${req.user!.email}`);
    res.json({ success: true });
  });

  app.post("/api/admin/users/:id/suspend", requireAuth, requireAdmin, (req: Request, res: Response) => {
    const user = db.getUserById(req.params.id);
    if (!user) return res.status(404).json({ error: "User not found" });
    db.updateUserStatus(req.params.id, "suspended");
    console.log(`[Auth] User ${user.email} suspended by ${req.user!.email}`);
    res.json({ success: true });
  });
}
