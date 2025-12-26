# GitHub Organization - Frequently Asked Questions (FAQ)

## Getting Started with Organizations

### Q: I paid for a GitHub organization, but where is it?

**A:** After paying for an organization:
1. Check your email for confirmation and setup instructions
2. Go to https://github.com/settings/organizations
3. Your organization should appear in the list
4. Click on it to access the organization page

---

### Q: How do I access my organization after creating it?

**A:** There are several ways:
1. **Direct URL**: `https://github.com/YOUR-ORG-NAME`
2. **GitHub Dashboard**: Click organization name in left sidebar
3. **Profile Menu**: Click your avatar → Select "Your organizations"
4. **Settings**: https://github.com/settings/organizations

---

## Repository Transfer Issues

### Q: I transferred repositories to my organization, but I can't see them. Why?

**A:** Common reasons:
1. **Not logged in as organization member**: Verify you're a member
2. **Filtering by visibility**: Make sure "All" is selected in repository filters
3. **No access permissions**: Organization owner needs to grant you access
4. **Transfer failed**: Check if repositories are still in your personal account
5. **Private repository limits**: Check if you hit the plan limit

**Solution**: See our comprehensive [Troubleshooting Guide](REPOSITORY_VISIBILITY_TROUBLESHOOTING.md)

---

### Q: How do I verify a repository transfer was successful?

**A:**
1. Check your personal repositories: `https://github.com/YOUR-USERNAME?tab=repositories`
   - The repository should NO LONGER be there
2. Check organization repositories: `https://github.com/orgs/YOUR-ORG-NAME/repositories`
   - The repository SHOULD be there
3. If still in personal account, the transfer didn't complete

---

### Q: Can I transfer repositories back to my personal account?

**A:** Yes! Organization owners or repository admins can:
1. Go to repository Settings
2. Scroll to Danger Zone
3. Click "Transfer repository"
4. Enter your personal username
5. Confirm the transfer

---

## Organization Membership

### Q: I created an organization, but I'm not a member?

**A:** When you create an organization:
1. You're automatically the owner (not just a member)
2. Go to `https://github.com/orgs/YOUR-ORG-NAME/people` to see your role
3. Your username should appear with "Owner" badge

---

### Q: I paid for an organization but didn't get an invitation. What should I do?

**A:**
1. **If you created the org**: You're automatically the owner, no invitation needed
2. **If someone else created it**: Contact the organization owner to send you an invitation
3. **Check spam folder**: GitHub invitations might be in spam
4. **Direct invitation link**: Ask owner to send you `https://github.com/orgs/ORG-NAME/invitation`

---

### Q: How do I invite others to my organization?

**A:**
1. Go to `https://github.com/orgs/YOUR-ORG-NAME/people`
2. Click "Invite member"
3. Enter their GitHub username or email
4. Select their role (Member or Owner)
5. Click "Send invitation"

---

## Permissions and Access

### Q: What's the difference between organization owner and member?

**A:**
| Feature | Owner | Member |
|---------|-------|--------|
| Manage organization settings | ✅ | ❌ |
| Invite/remove members | ✅ | ❌ |
| Create repositories | ✅ | ✅ (if allowed) |
| Delete organization | ✅ | ❌ |
| Manage billing | ✅ | ❌ |
| Access all repositories | ✅ | ❌ (needs permission) |

---

### Q: I'm a member but can't see any repositories. Why?

**A:** Possible reasons:
1. **Base permissions set to "None"**: Owner needs to change this
2. **Not added to any team**: Ask owner to add you to appropriate teams
3. **Repository-specific permissions**: Need to be granted explicitly
4. **Organization invitation not accepted**: Check for pending invitation

---

### Q: How do I give someone access to a specific repository?

**A:**
1. Go to repository Settings
2. Click "Manage access" (or "Collaborators and teams")
3. Click "Invite teams or people"
4. Search for the user or team
5. Select their permission level:
   - **Read**: View and clone only
   - **Triage**: Read + manage issues/PRs
   - **Write**: Triage + push commits
   - **Maintain**: Write + manage settings (no delete)
   - **Admin**: Full access including delete

---

## Billing and Plans

### Q: I paid for an organization, but it still shows as Free. Why?

**A:**
1. Payment might be processing (can take a few minutes)
2. Check billing page: `https://github.com/organizations/YOUR-ORG-NAME/settings/billing`
3. Verify payment method was charged successfully
4. Contact GitHub support if issue persists

---

### Q: What's the difference between Free and paid organization plans?

**A:**
| Feature | Free | Team | Enterprise |
|---------|------|------|------------|
| Public repositories | Unlimited | Unlimited | Unlimited |
| Private repositories | Unlimited | Unlimited | Unlimited |
| Users | Unlimited | Priced per user | Priced per user |
| GitHub Actions minutes | 2,000/month | 3,000/month | 50,000/month |
| Advanced features | Limited | More tools | Full features |
| Support | Community | Email support | Priority support |

---

### Q: I hit my private repository limit. What can I do?

**A:** Options:
1. **Upgrade plan**: Get more or unlimited private repositories
2. **Make some repositories public**: Convert unnecessary private repos
3. **Delete unused repositories**: Remove old or test repositories
4. **Archive repositories**: Archive but keep (still counts toward limit)

---

## Common Workflows

### Q: What's the recommended setup for my organization?

**A:**
1. **Create the organization** and choose a clear name
2. **Set up billing** if you need paid features
3. **Configure base permissions**: Settings → Member privileges
4. **Create teams** for different groups (e.g., Frontend, Backend, DevOps)
5. **Invite members** and assign to teams
6. **Transfer or create repositories**
7. **Set up team access** to repositories
8. **Configure security settings**: Settings → Security

---

### Q: How do I organize repositories in my organization?

**A:** Best practices:
1. **Use clear naming conventions**: `project-component` (e.g., `app-frontend`, `app-backend`)
2. **Use topics/tags**: Tag repositories for easy filtering
3. **Create teams**: Organize people by project or role
4. **Use repositories**: Keep one repo per component or microservice
5. **Archive old projects**: Keep organization clean

---

## Troubleshooting

### Q: None of the solutions work. What should I do?

**A:**
1. **Wait 5-10 minutes**: Sometimes changes take time to propagate
2. **Clear browser cache**: Try incognito/private mode
3. **Try different browser**: Rule out browser-specific issues
4. **Check GitHub Status**: https://www.githubstatus.com/
5. **Contact GitHub Support**: https://support.github.com/contact
   - Provide: Organization name, repository names, your username, screenshots

---

### Q: Where can I find more help?

**A:**
- **Detailed Troubleshooting**: [REPOSITORY_VISIBILITY_TROUBLESHOOTING.md](REPOSITORY_VISIBILITY_TROUBLESHOOTING.md)
- **Quick Reference**: [GITHUB_ORG_QUICK_REFERENCE.md](GITHUB_ORG_QUICK_REFERENCE.md)
- **GitHub Docs**: https://docs.github.com/en/organizations
- **GitHub Community**: https://github.community/
- **GitHub Support**: https://support.github.com/

---

## Additional Resources

### Official GitHub Documentation
- [About organizations](https://docs.github.com/en/organizations/collaborating-with-groups-in-organizations/about-organizations)
- [Creating a new organization](https://docs.github.com/en/organizations/collaborating-with-groups-in-organizations/creating-a-new-organization-from-scratch)
- [Managing organization settings](https://docs.github.com/en/organizations/managing-organization-settings)
- [Organization billing](https://docs.github.com/en/billing/managing-billing-for-your-github-account)

### Video Tutorials
- [GitHub Skills - Organizations](https://skills.github.com/)
- [YouTube - GitHub Organizations Guide](https://www.youtube.com/github)

---

**Pro Tip**: Bookmark your organization page and key documentation pages for quick access!

**Last Updated**: December 26, 2024
