# FROZEN COPY (round 9): hydrophysics_analyst/router/zh.py as at commit fc7b996, vendored for the
# R10.3b round-9 replay (hydrophysics_analyst.replay_r9). Do not edit: only the relative imports
# changed (one more dot). tests/test_analyst_r9_replay.py checks it still decides as fc7b996 did.
"""Simplified -> traditional Chinese for routing (the page and every pattern are traditional).

A fixed character map, not a general converter: it covers the page's township names (鄉/乡,
鎮/镇, 崙/仑, 庫/库, 東/东, 長/长 ...), the lever and class words (減/减, 養殖/养殖, 工業/工业,
槓桿/杠杆, 縣/县, 雲林/云林), and the words the rules and refusal guards read (head, subsidence,
time, money, observations, how-sure wording, the both-counties markers); a test folds those
words from their simplified forms. ``臺`` is folded to ``台`` as the page writes 台西鄉. Characters whose
simplified form is also a common traditional character with another meaning are left alone
(里, 干, 只, 系, 制, 范, 松, 余, 台).
"""
from __future__ import annotations

_PAIRS = (
    "减減 养養 渔漁 鱼魚 业業 杠槓 乡鄉 镇鎮 县縣 云雲 长長 头頭 东東 势勢 内內 麦麥 仑崙 库庫 "
    "农農 厂廠 场場 层層 盘盤 压壓 浊濁 积積 冲沖 区區 个個 两兩 会會 么麼 样樣 观觀 测測 监監 "
    "准準 验驗 证證 检檢 试試 为為 应應 该該 议議 费費 预預 钱錢 吨噸 铁鐵 车車 线線 后後 间間 "
    "来來 过過 这這 将將 几幾 数數 万萬 亿億 时時 门門 槛檻 计計 围圍 变變 弹彈 现現 开開 关關 "
    "环環 资資 补補 价價 额額 经經 济濟 级級 设設 实實 际際 质質 达達 对對 们們 让讓 说說 请請 "
    "问問 给給 诉訴 显顯 吗嗎 号號 点點 读讀 侧側 边邊 邻鄰 历歷 统統 卫衛 导導 兴興 坛壇 园園 "
    "员員 盐鹽 总總 处處 带帶 结結 报報 纪紀 国國 华華 产產 猪豬 饮飲 选選 择擇 种種 类類 别別 "
    "电電 单單 较較 马馬 启啟 动動 迟遲 继繼 续續 偿償 块塊 灾災 损損 稳穩 准準 确確 "
    "宽寬 广廣 务務 规規 则則 统統 计計 图圖 页頁 览覽 网網 络絡 级級 围圍 "
    "异異 匀勻 据據 评評 录錄 没沒 泄洩 缓緩 体體 输輸 与與 币幣 顷頃 还還 拟擬 换換 从從 采採 进進 "
    "杆桿 赖賴 释釋 认認 "
    "臺台"
)
S2T = str.maketrans({p[0]: p[1] for p in _PAIRS.split() if len(p) == 2})


def to_trad(s: str) -> str:
    return s.translate(S2T)
