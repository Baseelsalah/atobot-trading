import { Octokit } from '@octokit/rest';
import * as fs from 'fs';
import * as path from 'path';

let connectionSettings: any;

async function getAccessToken() {
  const hostname = process.env.REPLIT_CONNECTORS_HOSTNAME;
  const xReplitToken = process.env.REPL_IDENTITY 
    ? 'repl ' + process.env.REPL_IDENTITY 
    : process.env.WEB_REPL_RENEWAL 
    ? 'depl ' + process.env.WEB_REPL_RENEWAL 
    : null;

  if (!xReplitToken) throw new Error('X_REPLIT_TOKEN not found');

  connectionSettings = await fetch(
    'https://' + hostname + '/api/v2/connection?include_secrets=true&connector_names=github',
    { headers: { 'Accept': 'application/json', 'X_REPLIT_TOKEN': xReplitToken } }
  ).then(res => res.json()).then(data => data.items?.[0]);

  return connectionSettings?.settings?.access_token || connectionSettings.settings?.oauth?.credentials?.access_token;
}

// Only include essential source directories
const INCLUDE_DIRS = ['client', 'server', 'shared', 'scripts', 'db'];
const INCLUDE_ROOT_FILES = [
  '.gitignore', 'package.json', 'tsconfig.json', 'vite.config.ts', 
  'tailwind.config.ts', 'postcss.config.js', 'drizzle.config.ts',
  'replit.md', 'README.md', 'theme.json'
];

function getAllFiles(dir: string, baseDir: string = dir): string[] {
  const files: string[] = [];
  if (!fs.existsSync(dir)) return files;
  
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relativePath = path.relative(baseDir, fullPath);
    
    if (entry.isDirectory()) {
      if (!['node_modules', '.git', 'dist', '.cache'].includes(entry.name)) {
        files.push(...getAllFiles(fullPath, baseDir));
      }
    } else if (entry.isFile()) {
      files.push(relativePath);
    }
  }
  return files;
}

async function main() {
  const accessToken = await getAccessToken();
  const octokit = new Octokit({ auth: accessToken });
  
  const { data: user } = await octokit.users.getAuthenticated();
  const owner = user.login;
  const repo = 'atobot-trading';
  const baseDir = '/home/runner/workspace';
  
  console.log(`Pushing essential files to ${owner}/${repo}...`);
  
  // First initialize the repo with a README
  try {
    await octokit.repos.createOrUpdateFileContents({
      owner, repo,
      path: 'README.md',
      message: 'Initialize repository',
      content: Buffer.from('# AtoBot Trading Dashboard\n\nAI-powered stock trading dashboard with Alpaca API integration.\n').toString('base64')
    });
    console.log('Initialized repo with README');
  } catch (e: any) {
    if (e.status !== 422) console.log('README init:', e.message);
  }
  
  // Wait a moment for repo to be ready
  await new Promise(r => setTimeout(r, 2000));
  
  // Get all files from essential directories
  let files: string[] = [];
  for (const dir of INCLUDE_DIRS) {
    const dirPath = path.join(baseDir, dir);
    const dirFiles = getAllFiles(dirPath, baseDir);
    files.push(...dirFiles);
  }
  
  // Add root config files
  for (const file of INCLUDE_ROOT_FILES) {
    if (fs.existsSync(path.join(baseDir, file))) {
      files.push(file);
    }
  }
  
  console.log(`Found ${files.length} essential files`);
  
  // Create blobs
  const tree: { path: string; mode: '100644'; type: 'blob'; sha: string }[] = [];
  let uploaded = 0;
  
  for (const file of files) {
    const filePath = path.join(baseDir, file);
    try {
      const content = fs.readFileSync(filePath);
      const { data: blob } = await octokit.git.createBlob({
        owner, repo,
        content: content.toString('base64'),
        encoding: 'base64'
      });
      tree.push({ path: file, mode: '100644', type: 'blob', sha: blob.sha });
      uploaded++;
      if (uploaded % 20 === 0) console.log(`Uploaded ${uploaded}/${files.length}`);
    } catch (e: any) {
      console.error(`Failed: ${file} - ${e.message}`);
      if (e.message.includes('rate limit')) {
        console.log('Rate limited - waiting 60s...');
        await new Promise(r => setTimeout(r, 60000));
      }
    }
  }
  
  console.log(`\nCreating tree with ${tree.length} files...`);
  
  // Get current commit SHA
  const { data: ref } = await octokit.git.getRef({ owner, repo, ref: 'heads/main' });
  const parentSha = ref.object.sha;
  
  // Create tree
  const { data: newTree } = await octokit.git.createTree({
    owner, repo,
    base_tree: parentSha,
    tree: tree as any
  });
  
  // Create commit
  const { data: commit } = await octokit.git.createCommit({
    owner, repo,
    message: 'Add AtoBot Trading Dashboard source code',
    tree: newTree.sha,
    parents: [parentSha]
  });
  
  // Update main branch
  await octokit.git.updateRef({
    owner, repo,
    ref: 'heads/main',
    sha: commit.sha
  });
  
  console.log(`\nSuccess! Code pushed to: https://github.com/${owner}/${repo}`);
}

main().catch(console.error);
