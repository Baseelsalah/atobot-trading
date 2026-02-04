import { Octokit } from '@octokit/rest';
import * as fs from 'fs';
import * as path from 'path';

let connectionSettings: any;

async function getAccessToken() {
  if (connectionSettings && connectionSettings.settings.expires_at && new Date(connectionSettings.settings.expires_at).getTime() > Date.now()) {
    return connectionSettings.settings.access_token;
  }
  
  const hostname = process.env.REPLIT_CONNECTORS_HOSTNAME;
  const xReplitToken = process.env.REPL_IDENTITY 
    ? 'repl ' + process.env.REPL_IDENTITY 
    : process.env.WEB_REPL_RENEWAL 
    ? 'depl ' + process.env.WEB_REPL_RENEWAL 
    : null;

  if (!xReplitToken) {
    throw new Error('X_REPLIT_TOKEN not found');
  }

  connectionSettings = await fetch(
    'https://' + hostname + '/api/v2/connection?include_secrets=true&connector_names=github',
    {
      headers: {
        'Accept': 'application/json',
        'X_REPLIT_TOKEN': xReplitToken
      }
    }
  ).then(res => res.json()).then(data => data.items?.[0]);

  const accessToken = connectionSettings?.settings?.access_token || connectionSettings.settings?.oauth?.credentials?.access_token;
  if (!connectionSettings || !accessToken) {
    throw new Error('GitHub not connected');
  }
  return accessToken;
}

const IGNORE_PATTERNS = [
  'node_modules', '.git', 'dist', '.cache', '.replit', 
  'replit.nix', '.config', 'attached_assets', '.upm',
  'package-lock.json', '*.log', 'tmp', '.local'
];

function shouldIgnore(filePath: string): boolean {
  const parts = filePath.split('/');
  return IGNORE_PATTERNS.some(pattern => {
    if (pattern.includes('*')) {
      const ext = pattern.replace('*', '');
      return filePath.endsWith(ext);
    }
    return parts.includes(pattern);
  });
}

function getAllFiles(dir: string, baseDir: string = dir): string[] {
  const files: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relativePath = path.relative(baseDir, fullPath);
    
    if (shouldIgnore(relativePath)) continue;
    
    if (entry.isDirectory()) {
      files.push(...getAllFiles(fullPath, baseDir));
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
  
  console.log(`Pushing to ${owner}/${repo}...`);
  
  const files = getAllFiles(baseDir);
  console.log(`Found ${files.length} files to upload`);
  
  // Create blobs for all files
  const tree: { path: string; mode: '100644'; type: 'blob'; sha: string }[] = [];
  
  for (const file of files) {
    const filePath = path.join(baseDir, file);
    const content = fs.readFileSync(filePath);
    const base64Content = content.toString('base64');
    
    try {
      const { data: blob } = await octokit.git.createBlob({
        owner, repo,
        content: base64Content,
        encoding: 'base64'
      });
      
      tree.push({
        path: file,
        mode: '100644',
        type: 'blob',
        sha: blob.sha
      });
      process.stdout.write('.');
    } catch (e: any) {
      console.error(`\nFailed to upload ${file}: ${e.message}`);
    }
  }
  
  console.log(`\nCreating tree with ${tree.length} files...`);
  
  // Create tree
  const { data: newTree } = await octokit.git.createTree({
    owner, repo,
    tree: tree as any
  });
  
  // Create commit
  const { data: commit } = await octokit.git.createCommit({
    owner, repo,
    message: 'Initial commit: AtoBot Trading Dashboard',
    tree: newTree.sha,
    parents: []
  });
  
  // Update main branch reference
  try {
    await octokit.git.createRef({
      owner, repo,
      ref: 'refs/heads/main',
      sha: commit.sha
    });
    console.log('Created main branch');
  } catch (e: any) {
    if (e.status === 422) {
      await octokit.git.updateRef({
        owner, repo,
        ref: 'heads/main',
        sha: commit.sha,
        force: true
      });
      console.log('Updated main branch');
    } else {
      throw e;
    }
  }
  
  console.log(`\nSuccess! Code pushed to: https://github.com/${owner}/${repo}`);
}

main().catch(console.error);
