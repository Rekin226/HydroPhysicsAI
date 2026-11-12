# HydroPhysicsAI 主機移轉與部署指南

本文件說明如何將 HydroPhysicsAI 完整移轉至另一台 Linux 主機，並以 Docker 架設網頁平台、
執行研究運算環境，以及設定對外 IP 或網域。所有指令皆在專案根目錄執行，除非另有註明。

本儲存庫為公開專案。新主機的 IP、帳號、主機金鑰等連線資訊一律寫入
`docs/GPU_SERVER.local.md`（已列入 gitignore），不得寫入本文件或任何提交的檔案。

---

## 0. 系統總覽

| 部分 | 內容 | 硬體 | 部署方式 |
|---|---|---|---|
| **A. 網頁平台** | 分析助理（聊天）與決策頁（`results/twin/twin_app.html`），由 `hydrophysics_analyst` 提供 | CPU 即可，約 1 GB 記憶體 | Docker 服務 `web`，可選擇搭配 `caddy`（HTTPS）或 `tunnel`（Cloudflare） |
| **B. 研究運算環境** | 數位孿生求解器、PhysicsNeMo、Warp、測試與資料處理 | NVIDIA GPU | Docker 服務 `compute`（掛載整個專案目錄），或 conda 環境 |

```
                   網際網路
                      │
        ┌─────────────┴──────────────┐
        │ 方案 A/B：公網 IP 80/443     │ 方案 C：Cloudflare Tunnel（僅對外連線）
        ▼                            ▼
   ┌─────────┐                 ┌───────────┐
   │  caddy  │                 │  tunnel   │
   └────┬────┘                 └─────┬─────┘
        └──────────┬─────────────────┘
                   ▼  docker 內部網路 web:7860
             ┌───────────┐        127.0.0.1:7860（本機測試）
             │    web    │◄──────────────────────────────
             └───────────┘
   ┌──────────────────────────────────────────────┐
   │ compute（GPU）：/work = 專案目錄（程式＋資料） │
   └──────────────────────────────────────────────┘
```

### 0.1 移轉相關檔案（`deploy/host/`）

| 檔案 | 用途 |
|---|---|
| `preflight.sh` | 舊主機執行：列出未提交或未推送的工作、僅存在本機的資料、執行中的工作 |
| `transfer.sh` | 舊主機執行：以 rsync 或 tar 傳送完整專案（含 `.git` 與所有 gitignore 資料） |
| `verify_host.sh` | 新主機執行：檢查 Docker、GPU、輸入資料、連接埠 |
| `docker-compose.yml` | 所有服務：`web`、`caddy`、`tunnel`、`tunnel-quick`、`compute` |
| `env.template` | 設定範本，複製為 `deploy/host/.env`（gitignore）後填寫 |
| `Caddyfile` | 反向代理與自動 HTTPS 設定 |
| `compute.Dockerfile` | 研究運算映像檔（重現原主機 conda 環境 `hydro`） |
| `requirements-gpu.lock.txt` | 原主機 `hydro` 環境的完整套件版本（`pip freeze`，Python 3.11.16，torch 2.11.0+cu128） |

網頁服務沿用 `deploy/twin_space/Dockerfile` 與 `launch.py`（原主機正在使用的同一組檔案），
映像檔由 GitHub 公開儲存庫的指定 commit 建置。

### 0.2 需要移轉的內容

程式碼已在 GitHub。以下內容僅存在於原主機磁碟，必須另外傳送：

| 路徑 | 約略大小 | 說明 |
|---|---|---|
| `.env` | < 1 KB | API 金鑰（Anthropic、Hugging Face） |
| `docs/*.local.md`、`data_fetch_api/*.local.json` | 小 | 僅限本機的筆記與設定 |
| `chou-shui-data/` | 370 MB | 濁水溪原始資料集 |
| `AMP_V2/data/` | 570 MB | AMP 資料快取 |
| `data_fetch_api/` | 5.4 GB | 廠商 API 參考資料與量測資料（永不提交） |
| `results/` | 41 GB | 資料發布版本、預先登錄（preregistration）、孿生模型輸出、決策頁 |
| `pi_meeting/` | 30 MB | 會議資料 |
| 其他 worktree（`~/HydroPhysicsAI-*`） | 約 1 GB | 各 worktree 的 gitignore 檔案：運算結果、`*.local.md` 交接筆記、本機資料表 |
| `~/hydro_scratch*`（專案目錄外） | 約 20 GB | 長時間運算的暫存輸出（選擇性） |

`transfer.sh` 的完整模式約 50 GB（加上 `--with-worktrees` 約 51 GB）；`--lite` 模式略過可重新產生的大型運算輸出
（`results/twin_forward`、`results/surrogate`、`results/twin_runs`），約 25 GB。
已登錄的測試會以雜湊值核對部分 `results/` 檔案，因此建議使用完整模式，只有在磁碟或頻寬
受限時才使用 `--lite`。

---

## 1. 新主機需求

| 項目 | 建議 |
|---|---|
| 作業系統 | Ubuntu 22.04 或 24.04 LTS（x86_64） |
| CPU / 記憶體 | 8 核心、32 GB 以上（僅網頁平台時 2 核心、4 GB 即可） |
| 磁碟 | 150 GB 以上可用空間（資料 50 GB、映像檔約 12 GB、運算輸出成長空間） |
| GPU（研究運算） | NVIDIA GPU，驅動程式 525 版以上（torch cu128 輪檔的需求） |
| 網路 | 可連線 GitHub、PyPI、download.pytorch.org、Docker Hub；對外服務需求見第 4.3 節 |

**GPU 架構注意事項。** 原主機為 Quadro RTX 6000（Turing，sm_75），無 bf16、FP8、
Flash Attention 2（詳見 `docs/GPU_SERVER.md`）。若新主機為 Ampere 或更新架構，上述功能可用，
但已登錄的結果是以 fp32/fp16 產生；變更數值精度屬於方法變更，須先記錄並經過同意。
新主機的硬體資訊（不含連線資訊）請更新至 `docs/GPU_SERVER.md` 第 1 節。

---

## 2. 舊主機：移轉前準備

### 2.1 盤點

```bash
bash deploy/host/preflight.sh
```

每個 worktree 一行，列出變更檔案數與其中「內容不在任何遠端分支上」的檔案數
（`content on no remote`）。後者為 0 的 worktree 標示 `ok`：其變更內容已以相同內容存在於
GitHub 某個分支，不需另外提交。標示 `[!]` 的項目處理方式如下：

- **變更內容不在任何遠端**（`content on no remote` 大於 0）：提交並推送到 GitHub。
  主要工作目錄的檔案也會由 `transfer.sh` 原樣複製。
- **其他 worktree 有 gitignore 檔案**：運算結果、`*.local.md` 等只存在該目錄，
  傳送時加上 `--with-worktrees`（見 2.3）。
- **只存在本機的分支**（commits on no remote）：一般以 `git push origin <分支名稱>` 推送。
  分支若含有不得公開的資料（例如自報告轉錄的資料表），保留在本機即可；所有分支都在 `.git`
  中，`transfer.sh` 會一併複製。推送前依 `CLAUDE.md` 的規則檢查內容。
- **stash**：會隨 `.git` 一起複製，但建議先轉成分支：`git stash branch <新分支名稱> stash@{0}`。

### 2.2 停止執行中的工作

`preflight.sh` 會列出 tmux 工作階段、GPU 行程與 Docker 容器。複製 `results/` 之前，
應等待或停止仍在寫入的運算工作，避免複製到寫到一半的檔案。網頁容器可以在移轉期間持續運作。

### 2.3 傳送

**方式一：rsync 直接傳送（建議，可中斷後續傳）。** 新主機須能以 SSH 登入，且目標的上層目錄
可由該帳號寫入（例如先在新主機執行 `sudo mkdir -p /srv && sudo chown "$USER" /srv`）。

```bash
bash deploy/host/transfer.sh --to <帳號>@<新主機位址>:/srv/HydroPhysicsAI --with-worktrees --dry-run
```

```bash
bash deploy/host/transfer.sh --to <帳號>@<新主機位址>:/srv/HydroPhysicsAI --with-worktrees
```

主要工作目錄複製至 `/srv/HydroPhysicsAI`，其他 worktree 複製至同一層
（`/srv/HydroPhysicsAI-r7` 等）。結束時會印出新主機上要執行的 `git worktree repair` 指令，
請記下（見 3.3）。再次執行同一指令只會傳送有變動的檔案，可用於移轉當天的最後一次同步。
加上 `--with-scratch` 會一併傳送 `~/hydro_scratch*`；加上 `--with-drive-backup` 會一併傳送
家目錄的雲端硬碟備份腳本與其設定（見 3.4）。

**方式二：封存檔（隨身硬碟或兩台主機無法直接連線時）。**

```bash
bash deploy/host/transfer.sh --out /mnt/usb/hydro_migration --with-worktrees
```

產生 `HydroPhysicsAI_<日期>.tar.zst`（未安裝 zstd 時為 `.tar.gz`）、
`worktrees_<日期>.tar.zst` 與 `SHA256SUMS_<日期>`。
在新主機上：

```bash
cd /mnt/usb/hydro_migration && sha256sum -c SHA256SUMS_*
```

```bash
sudo mkdir -p /srv && sudo chown "$USER" /srv && tar -C /srv -I zstd -xf /mnt/usb/hydro_migration/HydroPhysicsAI_*.tar.zst
```

```bash
tar -C /srv -I zstd -xf /mnt/usb/hydro_migration/worktrees_*.tar.zst
```

解開後主要工作目錄為 `/srv/HydroPhysicsAI`，其他 worktree 位於同一層。
使用 `--with-drive-backup` 時另有 `home_files_<日期>.tar.zst`，解開至新主機家目錄（見 3.4）。

### 2.4 機密資料

`.env` 會隨上述兩種方式一起傳送。若選擇另行傳送，只能使用 `scp` 或 rsync over SSH，
不得透過 git、電子郵件、聊天軟體或雲端共享連結傳送。抵達後執行 `chmod 600 .env`。
`--with-drive-backup` 傳送的 `~/.config/rclone/rclone.conf` 含有雲端硬碟的登入 token，處理方式與
`.env` 相同；封存檔方式下，存放 `home_files_<日期>.tar.zst` 的隨身硬碟應妥善保管。

---

## 3. 新主機：安裝

### 3.1 Docker Engine 與 Compose

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git rsync zstd tmux
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

登出並重新登入後，以 `docker run --rm hello-world` 確認。

### 3.2 GPU：驅動程式與 NVIDIA Container Toolkit（僅研究運算需要）

```bash
sudo ubuntu-drivers install
sudo reboot
```

重新開機後先執行 `nvidia-smi`，確認驅動程式版本在 525 以上。接著安裝 Container Toolkit：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

容器內能看到 GPU 即完成。每次重新開機後都應先執行 `nvidia-smi`；核心升級可能使驅動程式失效。

### 3.3 檢查專案與資料

```bash
cd /srv/HydroPhysicsAI
git worktree repair <transfer.sh 印出的 worktree 路徑>
git worktree prune
bash deploy/host/verify_host.sh
```

`git worktree repair`（git 2.29 以上，Ubuntu 22.04 內建 2.34）將複製過來的 worktree 重新連結到
新的主要工作目錄，必須在 `prune` 之前執行；`prune` 會刪除路徑已不存在的 worktree 紀錄，
先執行會使複製過來的 worktree 失去連結。未使用 `--with-worktrees` 時略過 `repair`。
以 `git worktree list` 確認結果。worktree 的絕對路徑隨新主機改變，筆記或記憶檔中引用的舊路徑
（例如 `~/HydroPhysicsAI-r7/docs/...`）需對應到新位置。
`verify_host.sh` 的 `FAIL` 項目須全部排除；`--` 為提示訊息。

新主機需要自己的 GitHub 推送權限：建立新的 SSH 金鑰並加入 GitHub 帳號，或安裝 `gh` 並執行
`gh auth login`。接著在 `docs/GPU_SERVER.local.md` 記錄新主機的連線資訊。

### 3.4 雲端硬碟備份（`drive_backup.sh`，選擇性）

原主機以家目錄的 `~/drive_backup.sh` 將整個家目錄壓縮後上傳至雲端硬碟（rclone）。
`transfer.sh --with-drive-backup` 傳送以下三個檔案至新主機家目錄的相同位置；新主機上已存在的同名
檔案保持不變：

| 檔案 | 用途 |
|---|---|
| `~/drive_backup.sh` | 備份腳本 |
| `~/.local/bin/rclone` | rclone 執行檔 |
| `~/.config/rclone/rclone.conf` | rclone 設定，含雲端硬碟登入 token（權限 600） |

上傳進度紀錄 `~/.drive_backup_state` 屬於原主機的封存檔，不會傳送。封存檔方式下，以
`tar -C ~ --skip-old-files -xf home_files_<日期>.tar.zst` 解開。

**首次在新主機執行前**：

1. **指定新的雲端資料夾。** 腳本預設的資料夾存放原主機的封存檔，封存檔名稱以家目錄項目命名，
   新主機上傳同名封存檔會取代原主機的版本。以環境變數指定新資料夾執行：

   ```bash
   REMOTE=<遠端名稱>:<新資料夾名稱> ./drive_backup.sh
   ```

   或直接修改腳本中 `REMOTE` 的預設值。
2. **確認 `SRC`。** 腳本中的 `SRC` 為原主機家目錄的絕對路徑；新主機帳號或家目錄不同時，改為新路徑。
3. **確認工具。** `zstd` 已由 3.1 的 apt 指令安裝；`bash deploy/host/verify_host.sh` 會檢查 rclone、
   rclone 設定與 zstd。以 `~/.local/bin/rclone lsd <遠端名稱>:` 確認 token 仍有效；失效時執行
   `rclone config reconnect <遠端名稱>:` 重新授權。
4. **在 tmux 中執行**，與原主機相同：`tmux new -s backup ./drive_backup.sh`。

---

## 4. 網頁平台部署

### 4.1 設定檔

```bash
cp deploy/host/env.template deploy/host/.env
chmod 600 deploy/host/.env
```

編輯 `deploy/host/.env`：

| 變數 | 說明 |
|---|---|
| `REPO_SHA` | 要建置的 commit。建議填入完整或短 SHA（例如 `git rev-parse --short origin/main` 的結果），以便日後回復；填 `main` 則建置當下的 main |
| `ANTHROPIC_API_KEY` | 從專案根目錄 `.env` 複製。留空時只提供範例問題與表單，不呼叫語言模型 |
| `ANALYST_DAILY_LLM_CAP` | 每日（UTC）語言模型呼叫上限，預設 300，用於控制費用 |
| `WEB_BIND`、`WEB_PORT` | `web` 在主機上的發布位址，預設 `127.0.0.1:7860`（僅本機） |
| `SITE_ADDRESS` | 方案 A/B 使用，見 4.3 |
| `TUNNEL_TOKEN` | 方案 C 具名通道使用，見 4.3 |

以下以 `C` 代表 compose 指令前綴，每次開新的 shell 時先設定：

```bash
C="docker compose -f deploy/host/docker-compose.yml --env-file deploy/host/.env"
```

### 4.2 啟動並在本機測試

```bash
$C up -d --build --wait web
```

```bash
curl -s http://127.0.0.1:7860/api/meta | head -c 200
```

```bash
$C logs web | tail -5
```

日誌應出現 `twin analyst on :7860 (page …, ledger …, llm on, cap 300/day)`。
`llm off` 表示 `ANTHROPIC_API_KEY` 未設定。

此時服務只在本機可見。若在個人電腦上預覽，可透過 SSH 轉送：

```bash
ssh -L 7860:127.0.0.1:7860 <帳號>@<新主機位址>
```

然後以瀏覽器開啟 `http://localhost:7860`。

### 4.3 對外存取

先判斷主機是否擁有公網 IP：

```bash
ip -4 addr show scope global | grep inet
```

```bash
curl -4 -s https://ifconfig.me; echo
```

第一個指令顯示網卡上的位址，第二個顯示外界看到的位址。兩者相同，表示主機直接擁有公網 IP
（方案 A 或 B）；不同或第一個為私有位址（`10.`、`172.16–31.`、`192.168.`），表示主機位於
NAT 或校園防火牆之後（方案 C，或請網管設定轉址後改用方案 A）。

| 方案 | 條件 | 網址 | HTTPS |
|---|---|---|---|
| **A. 公網 IP＋網域**（建議） | 公網 IP、可設定 DNS 的網域、80/443 可由外部連入 | `https://twin.example.org` | 自動（Let's Encrypt） |
| **B. 僅公網 IP** | 公網 IP、80 可由外部連入 | `http://<公網IP>` | 無 |
| **C1. Cloudflare 具名通道** | 網域託管於 Cloudflare（免費方案即可）；不需公網 IP 與入站連接埠 | `https://twin.example.org` | 由 Cloudflare 提供 |
| **C2. Cloudflare 快速通道** | 不需帳號、網域、公網 IP | `https://<隨機>.trycloudflare.com`（每次重啟都會改變） | 由 Cloudflare 提供 |

#### 方案 A：公網 IP＋網域（Caddy 自動 HTTPS）

1. **DNS**：在網域的 DNS 管理介面新增 A 紀錄，主機名稱（例如 `twin`）指向新主機公網 IP。
   以 `dig +short twin.example.org` 確認解析結果等於公網 IP。
2. **防火牆**：開放 80/tcp、443/tcp、443/udp。主機上的 ufw：

   ```bash
   sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw allow 443/udp && sudo ufw enable
   ```

   雲端主機（AWS、GCP、Azure 等）另須在安全群組（Security Group / 防火牆規則）開放相同連接埠；
   學校或機構網路須向網管申請開放入站 80/443。Let's Encrypt 透過 80 埠驗證網域，80 埠必須可達。
3. **設定**：`deploy/host/.env` 中 `SITE_ADDRESS=twin.example.org`（只填網域，不加 `https://`）。
4. **啟動**：

   ```bash
   $C --profile public up -d
   ```

   ```bash
   $C --profile public logs caddy | grep -iE "certificate obtained|error"
   ```

   看到 `certificate obtained successfully` 後，以 `https://twin.example.org` 開啟。憑證會自動續期，
   存放於 Docker volume `caddy-data`。

#### 方案 B：僅公網 IP（HTTP）

`deploy/host/.env` 設定 `SITE_ADDRESS=:80`，防火牆開放 80/tcp，然後：

```bash
$C --profile public up -d
```

以 `http://<公網IP>` 開啟。此方案沒有加密，僅適合短期展示；正式提供他人使用時請改用方案 A 或 C。

#### 方案 C1：Cloudflare 具名通道（固定網址，不需公網 IP）

1. 登入 Cloudflare Zero Trust 控制台 → **Networks → Tunnels → Create a tunnel** → 類型選
   **Cloudflared** → 命名（例如 `hydrophysics`）。
2. 在安裝步驟頁面複製 token（`eyJ…` 開頭的長字串），填入 `deploy/host/.env` 的 `TUNNEL_TOKEN`。
3. 在 **Public Hostname** 分頁新增：Subdomain `twin`、Domain 選擇託管於 Cloudflare 的網域、
   Service 類型 `HTTP`、URL `web:7860`。
4. 啟動：

   ```bash
   $C --profile tunnel up -d
   ```

   ```bash
   $C --profile tunnel logs tunnel | grep -i "registered tunnel connection"
   ```

此方案只使用對外連線，主機不需開放任何入站連接埠，也適用於 NAT 或校園網路。

#### 方案 C2：Cloudflare 快速通道（臨時網址）

```bash
$C --profile tunnel-quick up -d
```

```bash
$C --profile tunnel-quick logs tunnel-quick | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1
```

網址為隨機產生，容器或主機重啟後會改變，需重新通知使用者。原主機目前即使用此方式
（在 tmux 中執行 `cloudflared`）；新主機改由 compose 服務管理，隨 Docker 自動重啟。

#### 路由器轉址（NAT 環境改用方案 A）

若主機位於可自行管理的路由器之後，可在路由器設定 port forwarding：外部 TCP 80、443
（及 UDP 443）轉送至新主機內網 IP 的相同連接埠，並在 DNS 將網域指向路由器的公網 IP，
之後依方案 A 操作。內網 IP 應在路由器設定 DHCP 保留，避免變動。

### 4.4 安全事項

- **開放存取**：`launch.py` 不需登入即可使用（2026-11-09 的部署決定），每位訪客都是同一個匿名使用者。
  若需限制存取，可在 `Caddyfile` 的站台區塊加入基本驗證：

  ```
  basic_auth {
  	reviewer <以 `docker run --rm caddy:2.8.4 caddy hash-password` 產生的雜湊值>
  }
  ```

  修改後執行 `$C --profile public restart caddy`。
- **費用控制**：輸入的問題每題呼叫一次 Claude；範例問題與表單不呼叫。每日上限由
  `ANALYST_DAILY_LLM_CAP` 控制，超過上限後回覆會引導使用者改用範例問題或表單。
- **只發布必要的連接埠**：`WEB_BIND` 保持 `127.0.0.1`。Docker 發布的連接埠會繞過 ufw 規則，
  因此 `WEB_BIND=0.0.0.0` 會使 7860 埠直接暴露於外部，且沒有 HTTPS。
- **金鑰**：`deploy/host/.env` 與根目錄 `.env` 權限為 600。輪替金鑰時修改 `.env` 後執行
  `$C up -d web`（重建容器以載入新值）。
- **公開內容**：網頁映像檔只包含公開建置版本（彙總資料，無個別觀測井序列）。
  `*.local.html` 私有建置版本不得部署。
- **稽核紀錄**：分析助理的問答紀錄存放於 Docker volume `analyst-audit`，更新映像檔時保留。

### 4.5 更新與回復

```bash
git fetch origin && git rev-parse --short origin/main
```

將結果填入 `deploy/host/.env` 的 `REPO_SHA`，然後：

```bash
$C up -d --build --wait web
```

映像檔標籤即為 `twin-analyst:<REPO_SHA>`，舊版本仍保留在本機。回復時將 `REPO_SHA` 改回舊值並執行
`$C up -d web`（映像檔已存在，不需重建）。通道與 Caddy 不需重啟，網址維持不變。
`docker images twin-analyst` 可列出已建置的版本。

### 4.6 開機自動啟動

所有服務皆設定 `restart: unless-stopped`。確認 Docker 服務開機自動啟動即可：

```bash
sudo systemctl enable docker
```

主機重新開機後，先前以 `up -d` 啟動的服務會自動恢復。

### 4.7 切換與停用舊主機

1. 新主機服務確認正常（4.2 與 4.3 的檢查皆通過）。
2. 方案 A/C1：將 DNS 或通道的主機名稱指向新主機後，網址不變。方案 C2：將新網址提供給使用者。
3. 停止舊主機的網頁容器與通道（舊主機上 `docker stop twin-analyst`，並結束 tmux 中的
   `cloudflared`）。
4. 舊主機的資料在確認新主機完整運作，且 `verify_host.sh` 與測試皆通過之前，不得刪除。

---

## 5. 研究運算環境（GPU）

### 5.1 以 Docker 執行（建議）

`deploy/host/.env` 中設定 `HOST_UID`、`HOST_GID`（`id -u`、`id -g` 的結果）與 `SCRATCH_DIR`
（長時間運算輸出的目錄，需事先建立）。建置映像檔（首次約 10–20 分鐘，約 12 GB）：

```bash
$C --profile gpu build compute
```

進入運算環境（整個專案目錄掛載於 `/work`，`SCRATCH_DIR` 掛載於 `/scratch`）：

```bash
$C --profile gpu run --rm compute
```

容器內確認 GPU：

```bash
python -c "import torch, warp; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); warp.init()"
```

容器以主機使用者身分執行，寫入專案目錄的檔案擁有者與主機一致。程式碼與資料都來自掛載的目錄，
修改程式不需重建映像檔；只有 `requirements-gpu.lock.txt` 或 `pyproject.toml` 的相依套件變更時才需重建。

**長時間工作**放在主機的 tmux 中，以免 SSH 中斷時一併結束：

```bash
tmux new -s train
```

在 tmux 內執行 `$C --profile gpu run --rm compute python -m <模組> <參數>`，以 `Ctrl+B` 再按 `D`
離開，`tmux attach -t train` 回到工作階段。

**測試**（CPU，約 11 分鐘）：

```bash
$C --profile gpu run --rm compute pytest -q
```

**MPS 平行執行（進階）。** 原主機以 CUDA MPS 同時執行三個 GPU 工作。容器中使用 MPS 時，
在主機啟動 `nvidia-cuda-mps-control -d`，並於 `compute` 服務加入 volume
`/tmp/nvidia-mps:/tmp/nvidia-mps`（compose 已設定 `ipc: host`）。

### 5.2 不使用 Docker：conda 環境

```bash
curl -fsSLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b
~/miniforge3/bin/conda create -y -n hydro python=3.11
~/miniforge3/envs/hydro/bin/pip install -r deploy/host/requirements-gpu.lock.txt
~/miniforge3/envs/hydro/bin/pip install --no-deps -e .
```

之後以 `conda activate hydro` 使用，與原主機環境相同（見 `CLAUDE.md` 的 Environment 一節）。

---

## 6. 驗收清單

| # | 檢查 | 指令或方法 |
|---|---|---|
| 1 | 資料完整 | `verify_host.sh` 無 `FAIL`；封存檔方式另以 `sha256sum -c` 核對 |
| 2 | git 狀態 | `git status`、`git log -1` 與舊主機一致；`git fetch` 可連線 |
| 3 | 網頁本機 | `curl http://127.0.0.1:7860/api/meta` 回傳 JSON；日誌顯示 `llm on` |
| 4 | 網頁對外 | 從主機以外的網路（例如手機行動網路）開啟網址，送出一題範例問題與一題自行輸入的問題 |
| 5 | HTTPS | 方案 A/C：瀏覽器顯示有效憑證 |
| 6 | GPU | 容器內 `torch.cuda.is_available()` 為 `True` |
| 7 | 測試 | `pytest -q` 全數通過 |
| 8 | 重開機 | `sudo reboot` 後 `nvidia-smi` 正常、網頁服務自動恢復 |

---

## 7. 疑難排解

| 症狀 | 原因與處理 |
|---|---|
| `permission denied ... docker.sock` | 使用者未加入 `docker` 群組，或加入後尚未重新登入 |
| `web` 建置失敗於 `git fetch` | `REPO_SHA` 不存在於 GitHub（尚未推送），或主機無法連線 GitHub |
| 日誌出現 `BindingError` | 該 commit 的決策頁與 claims ledger 不一致；改用已合併至 main 的 commit |
| 日誌顯示 `llm off` | `deploy/host/.env` 的 `ANTHROPIC_API_KEY` 為空；填入後執行 `$C up -d web` |
| Caddy 無法取得憑證 | DNS 尚未指向本機，或 80 埠未由外部開放（雲端安全群組、機構防火牆） |
| 外部無法連線但本機正常 | 防火牆或安全群組未開放；或主機位於 NAT 之後（以 4.3 的兩個指令比對 IP），改用方案 C |
| 快速通道網址失效 | 容器或主機重啟後網址會改變，以 4.3 方案 C2 的指令取得新網址；需要固定網址時改用方案 C1 |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit 未安裝或未設定，見 3.2 |
| 容器內 `torch.cuda.is_available()` 為 `False` | 驅動程式低於 525，或主機 `nvidia-smi` 失效（重開機後驅動程式需重新安裝） |
| 寫入專案目錄出現 `Permission denied` | `HOST_UID`、`HOST_GID` 與專案目錄擁有者不符 |
| 磁碟空間不足 | `docker system df` 檢視映像檔用量，`docker image prune` 清除未使用的映像檔 |
