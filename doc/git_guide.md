# Git 版本管理入门教程

> 适用项目：NPU_prj  
> 工具：Git + GitHub  
> 难度：⭐☆☆☆☆（零基础友好）

> 项目工程状态以 [current_status.md](current_status.md) 和 [task_breakdown.md](task_breakdown.md) 为准；本文只说明 Git 协作流程，不作为架构或验证结论。

---

## 目录

1. [Git 是什么，为什么要用它](#1-git-是什么为什么要用它)
2. [安装与初次配置](#2-安装与初次配置)
3. [克隆（Clone）项目到本地](#3-克隆clone项目到本地)
4. [日常工作流：拉取最新代码](#4-日常工作流拉取最新代码)
5. [创建自己的开发分支](#5-创建自己的开发分支)
6. [提交你的修改](#6-提交你的修改)
7. [将改动推送到远端](#7-将改动推送到远端)
8. [合并主分支（Merge / Rebase）](#8-合并主分支merge--rebase)
9. [⚠️ 合并禁忌与安全操作规范](#9-️-合并禁忌与安全操作规范)
10. [常用命令速查卡](#10-常用命令速查卡)
11. [常见错误与解决方法](#11-常见错误与解决方法)

---

## 1. Git 是什么，为什么要用它

Git 是一个**版本控制系统**，可以理解为你代码的"时光机"：

- 每次"存档"（commit）都能记录你改了什么、为什么改
- 多人协作时，大家互不干扰、各自在自己的"分支"上干活
- 出了问题随时回退到任意历史版本
- 配合 GitHub，代码还能备份到云端

```
本地仓库 ←→ 远端仓库（GitHub）
   ↕
工作目录（你的文件）
```

---

## 2. 安装与初次配置

### 2.1 安装 Git

- **Windows**：下载 [Git for Windows](https://git-scm.com/download/win)，全程默认安装即可
- **验证安装**：打开 PowerShell，输入：

```powershell
git --version
# 应输出类似：git version 2.44.0.windows.1
```

### 2.2 配置身份（只需做一次）

Git 的每次提交都会记录是谁提交的，所以先告诉 Git 你是谁：

```powershell
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱@example.com"
```

> 对应 GitHub 账号就用注册 GitHub 时的邮箱，这样提交记录才能和账号关联。

### 2.3 配置 GitHub SSH 密钥（推荐，可免密推送）

```powershell
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
# 一路回车即可，密钥保存在 C:\Users\你的用户名\.ssh\id_ed25519.pub

# 复制公钥内容
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | clip

# 然后去 GitHub → Settings → SSH and GPG keys → New SSH key，粘贴进去
```

测试是否配置成功：

```powershell
ssh -T git@github.com
# 成功会看到：Hi 用户名! You've successfully authenticated...
```

---

## 3. 克隆（Clone）项目到本地

"克隆"就是把远端仓库完整下载到你的电脑。

```powershell
# 进入你想存放项目的目录
cd D:\

# 克隆项目（把 URL 换成你的仓库地址）
git clone git@github.com:demonk69/NPU_prj.git

# 克隆完成后进入项目目录
cd NPU_prj
```

> 如果没有 SSH Key，也可以用 HTTPS 地址克隆：
> ```powershell
> git clone https://github.com/demonk69/NPU_prj.git
> ```

克隆完成后，你的本地目录就是一个完整的 Git 仓库，带有全部历史记录。

---

## 4. 日常工作流：拉取最新代码

每次开始工作前，**第一件事**就是同步远端的最新代码，避免和别人的改动冲突。

```powershell
# 确保你在 main 分支（主分支）
git checkout main

# 拉取远端最新代码
git pull origin main
```

命令解释：
| 命令 | 含义 |
|------|------|
| `git pull` | 相当于 `git fetch`（下载）+ `git merge`（合并）的组合 |
| `origin` | 远端仓库的别名（克隆时自动创建） |
| `main` | 要同步的分支名称 |

**养成好习惯**：每天开工的第一行命令就是 `git pull`。

---

## 5. 创建自己的开发分支

> **核心原则：永远不要直接在 `main` 分支上开发！**

`main` 分支是"干净的主线"，所有功能开发、bug 修复都应该在独立分支上进行。

### 5.1 创建并切换到新分支

```powershell
# 基于最新的 main 创建新分支
git checkout main
git pull origin main              # 先确保 main 是最新的！

# 创建新分支并立即切换过去
git checkout -b feature/pe-optimization
#              ↑ 分支名，建议用 feature/xxx 或 fix/xxx 的格式
```

### 5.2 分支命名建议

| 类型 | 格式示例 | 用途 |
|------|---------|------|
| 新功能 | `feature/fp16-support` | 添加新功能 |
| Bug 修复 | `fix/dma-overflow` | 修复已知问题 |
| 实验性 | `exp/new-arch` | 探索性改动，不保证合并 |
| 文档 | `docs/simulation-guide` | 仅改文档 |

### 5.3 查看当前在哪个分支

```powershell
git branch
# 当前分支前面会有 * 号

git status
# 第一行也会显示当前分支
```

### 5.4 在分支间切换

```powershell
git checkout main                  # 切回主分支
git checkout feature/pe-optimization  # 切回你的功能分支
```

---

## 6. 提交你的修改

在你的分支上做完修改后，按以下步骤"存档"。

### 6.1 查看改了什么

```powershell
git status         # 看哪些文件被修改了
git diff           # 看具体改了哪几行
```

### 6.2 添加到暂存区

```powershell
# 添加指定文件
git add rtl/pe_top.v

# 添加所有修改的文件（谨慎使用，确保没有多余文件）
git add .
```

> **注意**：`git add .` 会添加当前目录下所有改动，包括你可能不想提交的临时文件。  
> 建议先 `git status` 确认，再决定是否用 `.`。

### 6.3 提交（Commit）

```powershell
git commit -m "feat(pe): 优化 INT8 MAC 流水线，减少一级延迟"
```

提交信息格式建议（Conventional Commits）：

```
类型(范围): 简短描述

# 类型：
feat     新功能
fix      修复 bug
docs     文档变更
refactor 重构（不影响功能）
test     添加测试
chore    构建/配置相关
```

**好的提交信息**：
```
fix(dma): 修复 burst_len 越界导致 AXI 响应超时的问题
```

**糟糕的提交信息**（别这样）：
```
改了一些东西
update
aaa
```

---

## 7. 将改动推送到远端

本地提交后，推送到 GitHub 让其他人或备份能看到：

```powershell
# 第一次推送新分支时，需要设置上游
git push -u origin feature/pe-optimization

# 之后的推送可以简写
git push
```

推送后可以去 GitHub 仓库页面，会看到你的新分支，以及"Compare & pull request"的提示。

---

## 8. 合并主分支（Merge / Rebase）

当你的功能开发完成，需要把它合入 `main` 分支。有两种方式：

### 方式 A：通过 GitHub Pull Request（推荐）

这是最安全、最规范的合并方式，整个过程可追溯、可 Code Review：

1. 推送你的分支到 GitHub
2. 在 GitHub 仓库页面点击 **"Compare & pull request"**
3. 填写 PR 标题和描述（改了什么、为什么改）
4. 检查 diff，确认无误后点 **"Merge pull request"**

### 方式 B：本地合并（仅个人项目可用）

```powershell
# 1. 先更新 main
git checkout main
git pull origin main

# 2. 切到你的功能分支，先把最新 main 同步进来（减少冲突）
git checkout feature/pe-optimization
git rebase main          # 推荐用 rebase，保持提交历史干净

# 3. 处理完冲突（如果有）后，切回 main 合并
git checkout main
git merge feature/pe-optimization --no-ff    # --no-ff 保留合并记录

# 4. 推送到远端
git push origin main

# 5. 删除已合并的功能分支（可选）
git branch -d feature/pe-optimization
git push origin --delete feature/pe-optimization
```

### 处理合并冲突

当两个分支修改了同一文件的同一行，就会产生冲突：

```
<<<<<<< HEAD (main 分支的内容)
assign result = a + b;
=======
assign result = a + b + c;
>>>>>>> feature/pe-optimization (你分支的内容)
```

**解决步骤**：
1. 打开冲突文件，找到 `<<<<<<<` 标记
2. 手动选择保留哪段代码（或者两段都保留、融合）
3. 删除 `<<<<<<<`、`=======`、`>>>>>>>` 这三行标记
4. 保存文件，然后：

```powershell
git add 冲突文件名
git rebase --continue    # 如果是 rebase
# 或
git commit               # 如果是 merge
```

---

## 9. ⚠️ 合并禁忌与安全操作规范

> **以下操作可能造成不可恢复的代码丢失，务必认真阅读！**

---

### 🚫 禁止行为 #1：不经 Review 直接 push 到 main

```powershell
# ❌ 禁止！直接在 main 上开发并推送
git checkout main
# ... 修改代码 ...
git push origin main     # 绝对禁止未经 Review 直接推到主线
```

**为什么危险**：main 是所有人共用的基线，你一个未经验证的改动可能让所有人的代码崩溃。

---

### 🚫 禁止行为 #2：强制推送（force push）到公共分支

```powershell
# ❌ 极度危险！
git push --force origin main
git push -f origin main
```

**为什么危险**：`--force` 会**覆盖**远端历史，让其他人的本地仓库和远端不一致，可能导致他们的提交丢失。

> 唯一允许 force push 的场景：在**你自己的私有功能分支**上整理提交历史，且确认没有其他人在用这个分支。

---

### 🚫 禁止行为 #3：合并前不同步最新 main

```powershell
# ❌ 危险！直接合并一个落后很久的分支
git checkout main
git merge feature/old-branch    # 这个分支可能已经和 main 差了很多
```

**正确做法**：合并前先把 main 的最新改动同步到功能分支：

```powershell
git checkout feature/old-branch
git rebase main        # 先同步，再合并
```

---

### 🚫 禁止行为 #4：随意使用 `git reset --hard`

```powershell
# ❌ 危险！会丢弃所有未提交的改动，且无法撤销
git reset --hard HEAD
```

如果只是想暂时保存当前改动去做别的事，请用：

```powershell
git stash          # 安全地"存起来"
# 做完其他事后
git stash pop      # 取回来
```

---

### 🚫 禁止行为 #5：合并 RTL 综合结果或仿真日志到仓库

综合产生的网表（`.v`、`.sdf`）、仿真波形（`.vcd`、`.fst`）体积巨大且可再生，**不应纳入版本控制**。

确保 `.gitignore` 中有以下内容：

```gitignore
# 仿真输出
*.vcd
*.fst
*.lxt
sim_out/
waves/

# 综合结果
synth_out/
*.synth.v
*.sdf
```

---

### ✅ 安全操作 Checklist（合并前必查）

```
□ 1. 本分支是否已通过所有仿真测试？
□ 2. 是否已 git pull 同步了最新的 main？
□ 3. 提交信息是否清晰，能让人看懂你改了什么？
□ 4. 是否有大文件（>1MB）混进来了？（git status 检查）
□ 5. 是否修改了公共接口/顶层端口？（需要通知协作者）
□ 6. 是否有调试用的临时代码没删（如 $display、dummy assign）？
```

---

## 10. 常用命令速查卡

```powershell
# ─── 仓库管理 ───────────────────────────────────────────
git clone <url>               # 克隆远端仓库
git init                      # 在当前目录初始化新仓库

# ─── 分支操作 ───────────────────────────────────────────
git branch                    # 列出所有本地分支
git branch -a                 # 列出所有分支（包括远端）
git checkout -b <分支名>      # 创建并切换到新分支
git checkout <分支名>         # 切换到已有分支
git branch -d <分支名>        # 删除本地分支（已合并才能删）
git branch -D <分支名>        # 强制删除本地分支

# ─── 同步操作 ───────────────────────────────────────────
git fetch origin              # 拉取远端信息（不合并）
git pull origin main          # 拉取并合并 main 分支
git push origin <分支名>      # 推送到远端
git push -u origin <分支名>   # 首次推送，设置上游跟踪

# ─── 工作区操作 ─────────────────────────────────────────
git status                    # 查看当前状态
git diff                      # 查看未暂存的改动
git diff --staged             # 查看已暂存的改动
git add <文件>                # 添加文件到暂存区
git add .                     # 添加所有改动（谨慎）
git commit -m "消息"          # 提交
git stash                     # 临时保存未提交改动
git stash pop                 # 恢复临时保存的改动

# ─── 查看历史 ───────────────────────────────────────────
git log --oneline --graph     # 简洁图形化查看历史
git log -5                    # 查看最近 5 条提交
git show <commit-hash>        # 查看某次提交的内容

# ─── 撤销操作（小心使用！）─────────────────────────────
git restore <文件>            # 撤销工作区改动（未 add 的）
git restore --staged <文件>   # 从暂存区取消（不丢失改动）
git revert <commit-hash>      # 安全地撤销某次提交（生成新提交）
# ⚠️ 以下命令会丢失数据，使用前三思：
git reset --hard <commit>     # 回退到某次提交，丢弃之后所有改动
```

---

## 11. 常见错误与解决方法

### 错误 1：`fatal: not a git repository`

```
fatal: not a git repository (or any of the parent directories): .git
```

**原因**：当前目录不在 Git 仓库内。  
**解决**：`cd D:\NPU_prj` 进入仓库目录。

---

### 错误 2：推送被拒绝（rejected）

```
! [rejected] main -> main (non-fast-forward)
```

**原因**：远端有你本地没有的新提交，Git 不允许覆盖。  
**解决**：

```powershell
git pull --rebase origin main    # 先把远端改动同步进来
git push origin main             # 再推送
```

---

### 错误 3：合并冲突后不知道怎么办

**解决步骤**：

```powershell
git status                       # 查看哪些文件有冲突（显示 both modified）
# 用编辑器打开冲突文件，手动解决 <<<<<<< 标记
git add <解决好的文件>
git merge --continue             # 或 git rebase --continue
```

如果冲突太复杂想放弃，完全退出：

```powershell
git merge --abort
# 或
git rebase --abort
```

---

### 错误 4：`Permission denied (publickey)`

```
git@github.com: Permission denied (publickey).
```

**原因**：SSH 密钥未配置或未添加到 GitHub。  
**解决**：按第 2.3 节重新配置 SSH 密钥。

---

### 错误 5：不小心把大文件提交进去了

```powershell
# 从暂存区移除（还没 commit）
git restore --staged 大文件.vcd

# 已经 commit 但还没 push（撤销最后一次 commit，保留文件改动）
git reset HEAD~1

# 已经 push 了（复杂，谨慎操作）
git rm --cached 大文件.vcd
git commit -m "chore: 移除误提交的仿真波形文件"
git push origin <分支名>
```

然后把该文件类型加到 `.gitignore`！

---

## 总结：健康的 Git 工作流

```
每天开工：git pull origin main
    ↓
创建功能分支：git checkout -b feature/xxx
    ↓
小步快跑：频繁 git add + git commit（每完成一个小功能就提交一次）
    ↓
推送分支：git push origin feature/xxx
    ↓
在 GitHub 上发起 Pull Request → Code Review → 合并到 main
    ↓
删除已合并的分支（保持仓库整洁）
```

> **记住**：main 分支是神圣的，只有经过验证和 Review 的代码才能进入。
> 分支便宜，创建分支不需要任何代价，遇事不决先开分支。

---

*文档维护：NPU_prj 团队 | 最后更新：2026-04*
