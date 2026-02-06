module.exports = {
  apps: [
    {
      name: "atobot",
      script: "dist/index.cjs",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      // Restart every Monday-Friday at 4:00 AM ET (pre-market)
      cron_restart: "0 4 * * 1-5",
      error_file: "logs/err.log",
      out_file: "logs/out.log",
      merge_logs: true,
      time: true,
      env: {
        NODE_ENV: "production",
      },
      env_development: {
        NODE_ENV: "development",
      },
      // PM2 will restart the process if it crashes
      min_uptime: "10s",
      max_restarts: 10,
      restart_delay: 4000,
      // Graceful shutdown
      kill_timeout: 5000,
      listen_timeout: 3000,
      shutdown_with_message: true,
    },
  ],
};
