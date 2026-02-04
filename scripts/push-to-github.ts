import { Octokit } from '@octokit/rest';

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
    throw new Error('X_REPLIT_TOKEN not found for repl/depl');
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

async function main() {
  const accessToken = await getAccessToken();
  const octokit = new Octokit({ auth: accessToken });
  
  // Get authenticated user
  const { data: user } = await octokit.users.getAuthenticated();
  console.log(`Authenticated as: ${user.login}`);
  
  // Check if repo exists
  const repoName = 'atobot-trading';
  try {
    const { data: repo } = await octokit.repos.get({
      owner: user.login,
      repo: repoName
    });
    console.log(`Repository already exists: ${repo.html_url}`);
    console.log(`Clone URL: ${repo.clone_url}`);
  } catch (e: any) {
    if (e.status === 404) {
      // Create new repo
      console.log('Creating new repository...');
      const { data: newRepo } = await octokit.repos.createForAuthenticatedUser({
        name: repoName,
        description: 'AtoBot AI-powered trading dashboard with Alpaca API integration',
        private: true,
        auto_init: false
      });
      console.log(`Created repository: ${newRepo.html_url}`);
      console.log(`Clone URL: ${newRepo.clone_url}`);
    } else {
      throw e;
    }
  }
  
  // Output the remote URL for git
  console.log(`\nGit remote URL: https://${accessToken}@github.com/${user.login}/${repoName}.git`);
}

main().catch(console.error);
