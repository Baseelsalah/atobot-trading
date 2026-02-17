import { useState, useEffect, useCallback } from "react";
import * as api from "../api";
import type { AdminUser } from "../api";
import { addToast } from "../hooks";
import { Users, CheckCircle, XCircle, Shield } from "lucide-react";

export default function AdminPanel() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);

  const loadUsers = useCallback(async () => {
    try {
      const data = await api.fetchAllUsers();
      setUsers(data);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
    const id = setInterval(loadUsers, 10000);
    return () => clearInterval(id);
  }, [loadUsers]);

  const handleAction = async (userId: string, action: "approve" | "reject" | "suspend") => {
    try {
      if (action === "approve") await api.approveUser(userId);
      else if (action === "reject") await api.rejectUser(userId);
      else await api.suspendUser(userId);
      addToast(`User ${action}d`, "success");
      loadUsers();
    } catch (err: any) {
      addToast(err.message || `Failed to ${action}`, "error");
    }
  };

  const statusBadge = (status: string) => {
    const colors: Record<string, string> = {
      pending: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
      approved: "bg-green-500/10 text-green-400 border-green-500/20",
      rejected: "bg-red-500/10 text-red-400 border-red-500/20",
      suspended: "bg-gray-500/10 text-gray-400 border-gray-500/20",
    };
    return (
      <span className={`px-2 py-0.5 text-xs font-medium rounded border ${colors[status] || colors.pending}`}>
        {status}
      </span>
    );
  };

  if (loading) {
    return <div className="text-center text-gray-400 py-12">Loading users...</div>;
  }

  const pending = users.filter(u => u.status === "pending").length;

  return (
    <div className="space-y-4">
      <div className="bg-white dark:bg-[#16161e] rounded-xl border border-gray-200 dark:border-gray-800 p-6">
        <div className="flex items-center gap-3 mb-4">
          <Users className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">User Management</h3>
          <span className="text-sm text-gray-500">{users.length} users</span>
          {pending > 0 && (
            <span className="px-2 py-0.5 text-xs font-medium rounded bg-yellow-500/10 text-yellow-400 border border-yellow-500/20">
              {pending} pending
            </span>
          )}
        </div>

        {users.length === 0 ? (
          <p className="text-gray-500 text-sm">No users registered yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b border-gray-800">
                  <th className="pb-2 font-medium">Email</th>
                  <th className="pb-2 font-medium">Name</th>
                  <th className="pb-2 font-medium">Role</th>
                  <th className="pb-2 font-medium">Status</th>
                  <th className="pb-2 font-medium">Registered</th>
                  <th className="pb-2 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                    <td className="py-3 text-white">{u.email}</td>
                    <td className="py-3 text-gray-300">{u.display_name}</td>
                    <td className="py-3">
                      {u.role === "admin" ? (
                        <span className="flex items-center gap-1 text-blue-400 text-xs">
                          <Shield className="w-3 h-3" /> Admin
                        </span>
                      ) : (
                        <span className="text-gray-400 text-xs">User</span>
                      )}
                    </td>
                    <td className="py-3">{statusBadge(u.status)}</td>
                    <td className="py-3 text-gray-500 text-xs">
                      {new Date(u.created_at).toLocaleDateString()}
                    </td>
                    <td className="py-3 text-right">
                      {u.role !== "admin" && (
                        <div className="flex gap-1 justify-end">
                          {(u.status === "pending" || u.status === "rejected" || u.status === "suspended") && (
                            <button onClick={() => handleAction(u.id, "approve")}
                              className="px-2 py-1 text-xs bg-green-600/20 text-green-400 rounded hover:bg-green-600/30 flex items-center gap-1">
                              <CheckCircle className="w-3 h-3" /> Approve
                            </button>
                          )}
                          {u.status === "pending" && (
                            <button onClick={() => handleAction(u.id, "reject")}
                              className="px-2 py-1 text-xs bg-red-600/20 text-red-400 rounded hover:bg-red-600/30 flex items-center gap-1">
                              <XCircle className="w-3 h-3" /> Reject
                            </button>
                          )}
                          {u.status === "approved" && (
                            <button onClick={() => handleAction(u.id, "suspend")}
                              className="px-2 py-1 text-xs bg-orange-600/20 text-orange-400 rounded hover:bg-orange-600/30">
                              Suspend
                            </button>
                          )}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
