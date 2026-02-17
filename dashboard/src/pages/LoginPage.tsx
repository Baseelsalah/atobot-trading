import { useState } from "react";

interface LoginPageProps {
  onLogin: (email: string, password: string) => Promise<any>;
  onRegister: (email: string, password: string, displayName: string) => Promise<any>;
}

export default function LoginPage({ onLogin, onRegister }: LoginPageProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pendingApproval, setPendingApproval] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === "register") {
        if (password !== confirmPassword) {
          setError("Passwords do not match");
          setLoading(false);
          return;
        }
        if (password.length < 6) {
          setError("Password must be at least 6 characters");
          setLoading(false);
          return;
        }
        const result = await onRegister(email, password, displayName);
        if (result.user.status === "pending") {
          setPendingApproval(true);
        }
      } else {
        await onLogin(email, password);
      }
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  if (pendingApproval) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center p-4">
        <div className="w-full max-w-md bg-[#16161e] rounded-xl border border-gray-800 p-8 text-center">
          <div className="text-4xl mb-4">&#9203;</div>
          <h2 className="text-xl font-bold text-white mb-2">Account Created</h2>
          <p className="text-gray-400 mb-6">
            Your account is pending admin approval. You'll be able to log in once an admin approves your account.
          </p>
          <button
            onClick={() => { setPendingApproval(false); setMode("login"); }}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Back to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-[#16161e] rounded-xl border border-gray-800 p-8">
        {/* Logo */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-white">AtoBot</h1>
          <p className="text-gray-500 text-sm mt-1">AI-Powered Trading Platform</p>
        </div>

        {/* Mode Toggle */}
        <div className="flex mb-6 bg-[#0a0a0f] rounded-lg p-1">
          <button
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
              mode === "login" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"
            }`}
            onClick={() => { setMode("login"); setError(null); }}
          >
            Login
          </button>
          <button
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
              mode === "register" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"
            }`}
            onClick={() => { setMode("register"); setError(null); }}
          >
            Create Account
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === "register" && (
            <div>
              <label className="block text-sm text-gray-400 mb-1">Display Name</label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                required
                className="w-full px-3 py-2 bg-[#0a0a0f] border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                placeholder="Your name"
              />
            </div>
          )}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-3 py-2 bg-[#0a0a0f] border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-3 py-2 bg-[#0a0a0f] border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              placeholder={mode === "register" ? "Min 6 characters" : "Enter password"}
            />
          </div>
          {mode === "register" && (
            <div>
              <label className="block text-sm text-gray-400 mb-1">Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                className="w-full px-3 py-2 bg-[#0a0a0f] border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                placeholder="Confirm password"
              />
            </div>
          )}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
          </button>
        </form>
      </div>
    </div>
  );
}
