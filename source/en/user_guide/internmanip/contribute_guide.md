# 🤝 Contribute Guide

We welcome all kinds of contributions to **InternManip** — not just code! Whether you’re improving documentation, reporting issues, adding new datasets or policies, or simply helping others in the community, your effort is highly valued and appreciated.



## 🚀 Ways to Contribute

There are many ways to get involved with InternManip:

- 🔧 Fix bugs or address issues in the codebase.
- 🧠 Implement new manipulation models or dataset loaders.
- 🧪 Add new simulation benchmarks or evaluation tasks.
- 📚 Improve the documentation or add tutorials and usage examples.
- 🐞 Report bugs and suggest new features via GitHub Issues.
- 💬 Reach out on our [Discord](https://discord.gg/5jeaQHUj4B) or via [WeChat](https://cdn.vansin.top/taoyuan.jpg) to discuss contributions or get help.

If you’re not sure how to contribute, feel free to open an issue to ask questions or discuss ideas!


## 📝 Submitting Issues or Feature Requests

### 🐛 Did You Find a Bug?

Thanks for helping us improve InternManip! To report a bug:

1. First, check if the issue has already been reported via the [GitHub Issues](https://github.com/InternRobotics/InternManip/issues) page.
2. If not, create a new issue and include:
   - Your OS, Python version, and PyTorch version.
   - A short code snippet that reproduces the bug (ideally under 30s).
   - The full error traceback.
   - Any other useful information (e.g., screenshots).

### ✨ Want to Request a New Feature?

When requesting a new feature, please try to:

- Explain the **motivation**: what problem are you solving or what project needs it?
- Describe the **proposed feature** and its impact.
- Include code snippets, diagrams, or links to papers (if applicable).


## 🧩 Adding a New Dataset, Model, or Benchmark

We encourage contributions that extend InternManip to new domains!
Below are quick links to common scenarios:


- ✍🏻 [How to customize your own model?](quick_start/add_model.md)
- 📦 [How to import a new dataset?](quick_start/add_dataset.md)
- 🥇 [How to add a new benchmark?](quick_start/add_benchmark.md)

We warmly welcome you to explore, contribute, and share your feedback. Feel free to open issues or reach out with suggestions!


## 🔁 Submitting a Pull Request (PR)

### 1. Fork and Clone

```bash
git clone git@github.com:<your-username>/internmanip.git
cd internmanip
git remote add upstream https://github.com/InternRobotics/InternManip.git
```
### 2. Create a Branch
```bash
git checkout -b my-feature-branch
```
### 3. Add Your Feature and Run Tests
```bash
"your develop and test code"
```
### 4. Format and Commit
```bash
pre-commit install
pre-commit run --all-files

git add .
git commit -m "Add new manipulation policy XYZ"
```
### 5. Push and Open PR
```bash
git push origin my-feature-branch
```

## ✅ PR Checklist
- [ ] PR title summarizes the change.
- [ ] New tests are included.
- [ ] All tests pass locally.
- [ ] Code is formatted via pre-commit.
- [ ] Issue number is referenced (if applicable).

## 💬 Get in Touch

We love connecting with contributors! You can reach out via:
- 💬 [Discord](https://discord.gg/5jeaQHUj4B): Join our community.
- 🌐 [WeChat](https://cdn.vansin.top/taoyuan.jpg): Scan our QR code in the README.
- 🐛 GitHub Issues: Ask questions or propose features.

## ⭐️ Show Support
If you find InternManip useful in your research or work:
- Star the repo ⭐️.
- Mention us in blog posts or papers.
- Share your projects with us!
