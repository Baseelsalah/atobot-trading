import { useState, useEffect } from "react";
import * as api from "../api";
import { addToast } from "../hooks";
import { Key, CheckCircle, XCircle, Loader2 } from "lucide-react";

export default function ApiKeySettings() {
  const [hasKeys, setHasKeys] = useState(false);
  const [isPaper, setIsPaper] = useState(true);
  const [alpacaKey, setAlpacaKey] = useState("");
  const [alpacaSecret, setAlpacaSecret] = useState("");
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; equity?: string; error?: string } | null>(null);

  useEffect(() => {
    api.fetchApiKeyStatus().then((status) => {
      setHasKeys(status.hasKeys);
      setIsPaper(status.isPaper);
    }).catch(() => {});
  }, []);

  const handleSave = async () => {
    if (!alpacaKey || !alpacaSecret) {
      addToast("Enter both API key and secret", "error");
      return;
    }
    setSaving(true);
    try {
      await api.saveApiKeys(alpacaKey, alpacaSecret, isPaper);
      setHasKeys(true);
      setAlpacaKey("");
      setAlpacaSecret("");
      addToast("API keys saved and encrypted", "success");
    } catch (err: any) {
      addToast(err.message || "Failed to save keys", "error");
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    if (!alpacaKey || !alpacaSecret) {
      addToast("Enter keys to test", "error");
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const result = await api.testApiKeys(alpacaKey, alpacaSecret, isPaper);
      setTestResult(result);
    } catch (err: any) {
      setTestResult({ success: false, error: err.message });
    } finally {
      setTesting(false);
    }
  };

  const handleRemove = async () => {
    if (!confirm("Remove your API keys? The bot will stop trading on your account.")) return;
    try {
      await api.deleteApiKeys();
      setHasKeys(false);
      addToast("API keys removed", "info");
    } catch (err: any) {
      addToast(err.message || "Failed to remove keys", "error");
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-[#16161e] rounded-xl border border-gray-200 dark:border-gray-800 p-6">
        <div className="flex items-center gap-3 mb-4">
          <Key className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Alpaca API Keys</h3>
          {hasKeys ? (
            <span className="ml-auto px-2 py-1 text-xs font-medium rounded bg-green-500/10 text-green-400 border border-green-500/20">
              Keys Configured
            </span>
          ) : (
            <span className="ml-auto px-2 py-1 text-xs font-medium rounded bg-red-500/10 text-red-400 border border-red-500/20">
              No Keys Set
            </span>
          )}
        </div>

        <p className="text-sm text-gray-500 mb-4">
          Enter your Alpaca API credentials. Keys are encrypted at rest and never displayed after saving.
        </p>

        {/* Paper/Live Toggle */}
        <div className="flex items-center gap-3 mb-4">
          <span className="text-sm text-gray-400">Mode:</span>
          <button
            onClick={() => setIsPaper(true)}
            className={`px-3 py-1 text-xs rounded-full ${isPaper ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-400"}`}
          >
            Paper
          </button>
          <button
            onClick={() => setIsPaper(false)}
            className={`px-3 py-1 text-xs rounded-full ${!isPaper ? "bg-orange-600 text-white" : "bg-gray-700 text-gray-400"}`}
          >
            Live
          </button>
        </div>

        <div className="space-y-3">
          <input
            type="password"
            value={alpacaKey}
            onChange={(e) => setAlpacaKey(e.target.value)}
            placeholder={hasKeys ? "Enter new key to update" : "APCA-API-KEY-ID"}
            className="w-full px-3 py-2 bg-[#0a0a0f] border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
          />
          <input
            type="password"
            value={alpacaSecret}
            onChange={(e) => setAlpacaSecret(e.target.value)}
            placeholder={hasKeys ? "Enter new secret to update" : "APCA-API-SECRET-KEY"}
            className="w-full px-3 py-2 bg-[#0a0a0f] border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
          />
        </div>

        {testResult && (
          <div className={`mt-3 p-3 rounded-lg text-sm ${testResult.success ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"}`}>
            {testResult.success ? (
              <span className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                Connected! Account equity: ${Number(testResult.equity).toLocaleString()}
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <XCircle className="w-4 h-4" />
                {testResult.error}
              </span>
            )}
          </div>
        )}

        <div className="flex gap-2 mt-4">
          <button onClick={handleTest} disabled={testing || !alpacaKey || !alpacaSecret}
            className="px-4 py-2 text-sm bg-gray-700 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 flex items-center gap-2">
            {testing ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
            Test Connection
          </button>
          <button onClick={handleSave} disabled={saving || !alpacaKey || !alpacaSecret}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {saving ? "Saving..." : "Save Keys"}
          </button>
          {hasKeys && (
            <button onClick={handleRemove}
              className="px-4 py-2 text-sm bg-red-600/20 text-red-400 rounded-lg hover:bg-red-600/30 ml-auto">
              Remove Keys
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
