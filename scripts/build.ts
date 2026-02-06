#!/usr/bin/env tsx
/**
 * Production Build Script
 *
 * Bundles the server code into a single CommonJS file for production deployment.
 * Uses esbuild for fast, efficient bundling.
 */

import { build } from "esbuild";
import fs from "fs";
import path from "path";

const distDir = path.join(process.cwd(), "dist");
const entryPoint = path.join(process.cwd(), "server/index.ts");

async function buildProduction() {
  console.log("🔨 Building production bundle...\n");

  // Clean dist directory
  if (fs.existsSync(distDir)) {
    fs.rmSync(distDir, { recursive: true });
    console.log("✅ Cleaned dist/ directory");
  }
  fs.mkdirSync(distDir, { recursive: true });

  // Build server bundle
  try {
    await build({
      entryPoints: [entryPoint],
      bundle: true,
      platform: "node",
      target: "node20",
      format: "cjs",
      outfile: path.join(distDir, "index.cjs"),
      external: [
        // Don't bundle node built-ins
        "fs",
        "path",
        "http",
        "https",
        "crypto",
        "stream",
        "zlib",
        "url",
        "net",
        "tls",
        "events",
        "util",
        // Don't bundle large dependencies that work better as externals
        "pg-native",
        "bufferutil",
        "utf-8-validate",
      ],
      sourcemap: true,
      minify: false, // Keep readable for debugging
      logLevel: "info",
      loader: {
        ".node": "file",
      },
    });

    console.log("✅ Server bundle created: dist/index.cjs");
  } catch (error) {
    console.error("❌ Build failed:", error);
    process.exit(1);
  }

  // Copy package.json (needed for production dependencies)
  const packageJson = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "package.json"), "utf-8")
  );

  // Create minimal package.json for production
  const prodPackageJson = {
    name: packageJson.name,
    version: packageJson.version,
    type: "commonjs",
    main: "index.cjs",
    dependencies: packageJson.dependencies,
  };

  fs.writeFileSync(
    path.join(distDir, "package.json"),
    JSON.stringify(prodPackageJson, null, 2),
    "utf-8"
  );
  console.log("✅ Copied package.json to dist/");

  // Create production .env.example
  const envExample = `# Production Environment Variables
# Copy this to .env and fill in your values

# Alpaca API
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret

# OpenAI API
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Trading Mode
DRY_RUN=0
TIME_GUARD_OVERRIDE=0

# Server
NODE_ENV=production
PORT=5000
`;

  fs.writeFileSync(path.join(distDir, ".env.example"), envExample, "utf-8");
  console.log("✅ Created .env.example in dist/");

  console.log("\n🎉 Build complete!\n");
  console.log("To run in production:");
  console.log("  1. cd dist/");
  console.log("  2. npm install --production");
  console.log("  3. cp .env.example .env (and configure)");
  console.log("  4. node index.cjs\n");
  console.log("Or use PM2:");
  console.log("  npm run pm2:start\n");
}

buildProduction().catch((error) => {
  console.error("❌ Build script crashed:", error);
  process.exit(1);
});
