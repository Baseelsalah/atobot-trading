#!/bin/sh
# Fix ownership of mounted volumes (they start as root)
if [ "$(id -u)" = "0" ]; then
    chown -R atobot:atobot /app/data /app/logs 2>/dev/null
    exec su-exec atobot "$@"
else
    exec "$@"
fi
