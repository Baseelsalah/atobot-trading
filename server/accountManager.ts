import crypto from "crypto";
import * as db from "./database";
import * as alpaca from "./alpaca";
import * as ato from "./ato";

const ALGO = "aes-256-gcm";

function getEncryptionKey(): Buffer {
  const key = process.env.ENCRYPTION_KEY;
  if (!key) throw new Error("ENCRYPTION_KEY environment variable not set");
  return Buffer.from(key, "hex");
}

export function encrypt(text: string): string {
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv(ALGO, getEncryptionKey(), iv);
  const encrypted = Buffer.concat([cipher.update(text, "utf8"), cipher.final()]);
  const tag = cipher.getAuthTag();
  return `${iv.toString("hex")}:${tag.toString("hex")}:${encrypted.toString("hex")}`;
}

export function decrypt(data: string): string {
  const [ivHex, tagHex, encHex] = data.split(":");
  const decipher = crypto.createDecipheriv(ALGO, getEncryptionKey(), Buffer.from(ivHex, "hex"));
  decipher.setAuthTag(Buffer.from(tagHex, "hex"));
  return Buffer.concat([decipher.update(Buffer.from(encHex, "hex")), decipher.final()]).toString("utf8");
}

export interface DecryptedUserAccount {
  userId: string;
  email: string;
  displayName: string;
  alpacaKey: string;
  alpacaSecret: string;
  isPaper: boolean;
}

export function getApprovedUsersWithDecryptedKeys(): DecryptedUserAccount[] {
  const users = db.getApprovedUsersWithKeys();
  const results: DecryptedUserAccount[] = [];

  for (const user of users) {
    try {
      results.push({
        userId: user.id,
        email: user.email,
        displayName: user.display_name,
        alpacaKey: decrypt(user.alpaca_key_encrypted!),
        alpacaSecret: decrypt(user.alpaca_secret_encrypted!),
        isPaper: !!user.is_paper,
      });
    } catch (err: any) {
      console.error(`[AccountManager] Failed to decrypt keys for ${user.email}: ${err.message}`);
    }
  }

  return results;
}

export function getUserDecryptedKeys(userId: string): { key: string; secret: string; isPaper: boolean } | null {
  const keys = db.getUserAlpacaKeys(userId);
  if (!keys) return null;

  try {
    return {
      key: decrypt(keys.encryptedKey),
      secret: decrypt(keys.encryptedSecret),
      isPaper: keys.isPaper,
    };
  } catch (err: any) {
    console.error(`[AccountManager] Failed to decrypt keys for user ${userId}: ${err.message}`);
    return null;
  }
}

export async function getUserEquity(alpacaKey: string, alpacaSecret: string): Promise<number | null> {
  alpaca.setActiveCredentials(alpacaKey, alpacaSecret);
  try {
    const account = await alpaca.getAccount();
    return parseFloat(account.equity);
  } catch (err: any) {
    console.error(`[AccountManager] Failed to get account: ${err.message}`);
    return null;
  } finally {
    alpaca.clearActiveCredentials();
  }
}

export function calculateScaledQuantity(baseQuantity: number, adminEquity: number, userEquity: number): number {
  if (adminEquity <= 0 || userEquity <= 0) return 0;
  return Math.floor(baseQuantity * (userEquity / adminEquity));
}

export interface MultiAccountResult {
  userId: string;
  email: string;
  success: boolean;
  scaledQty: number;
  message: string;
  orderId?: string;
}

export async function executeTradeForAllUsers(
  recommendation: any,
  baseQuantity: number,
  adminEquity: number,
  settings: any
): Promise<MultiAccountResult[]> {
  const users = getApprovedUsersWithDecryptedKeys();
  // Filter out admin (admin's env-var-based trade already executed)
  // We identify non-admin users by checking they have user role
  const nonAdminUsers = users.filter(u => {
    const dbUser = db.getUserById(u.userId);
    return dbUser && dbUser.role !== "admin";
  });

  if (nonAdminUsers.length === 0) return [];

  console.log(`[AccountManager] Executing trade for ${nonAdminUsers.length} user(s): ${recommendation.symbol} ${recommendation.side}`);

  const results: MultiAccountResult[] = [];

  for (const user of nonAdminUsers) {
    try {
      // Get user's equity
      const userEquity = await getUserEquity(user.alpacaKey, user.alpacaSecret);
      if (!userEquity || userEquity <= 0) {
        results.push({ userId: user.userId, email: user.email, success: false, scaledQty: 0, message: "Could not fetch account equity" });
        continue;
      }

      // Scale position
      const scaledQty = calculateScaledQuantity(baseQuantity, adminEquity, userEquity);
      if (scaledQty < 1) {
        results.push({ userId: user.userId, email: user.email, success: false, scaledQty: 0, message: `Account too small (${userEquity.toFixed(0)} vs ${adminEquity.toFixed(0)})` });
        continue;
      }

      // Switch credentials and execute
      alpaca.setActiveCredentials(user.alpacaKey, user.alpacaSecret);
      try {
        const scaledRec = { ...recommendation, quantity: scaledQty };
        const tradeResult = await ato.executeTraderDecision(scaledRec, settings);

        if (tradeResult.success) {
          db.logTrade(
            user.userId, recommendation.symbol, recommendation.side,
            scaledQty, recommendation.price, scaledQty * (recommendation.price || 0),
            recommendation.strategyName || null, "filled", tradeResult.orderId || null
          );
          results.push({ userId: user.userId, email: user.email, success: true, scaledQty, message: "Trade executed", orderId: tradeResult.orderId });
          console.log(`[AccountManager] ✓ ${user.email}: ${recommendation.side} ${scaledQty} ${recommendation.symbol}`);
        } else {
          results.push({ userId: user.userId, email: user.email, success: false, scaledQty, message: tradeResult.message || "Execution failed" });
          console.log(`[AccountManager] ✗ ${user.email}: ${tradeResult.message}`);
        }
      } finally {
        alpaca.clearActiveCredentials();
      }
    } catch (err: any) {
      alpaca.clearActiveCredentials();
      results.push({ userId: user.userId, email: user.email, success: false, scaledQty: 0, message: err.message });
      console.error(`[AccountManager] Error for ${user.email}: ${err.message}`);
    }
  }

  return results;
}
