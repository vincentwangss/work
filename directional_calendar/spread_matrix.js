/**
 * spread_matrix.js
 * 股指期货价差矩阵可视化 - 纯前端逻辑（无 Python 模板依赖）
 *
 * 依赖：
 *   - Chart.js (CDN)
 *   - 页面中需有 <script id="appData" type="application/json"> 注入的数据
 *   - 页面中需有 <script id="appConfig" type="application/json"> 注入的配置
 */

// ============================================================
// 常量（由 Python 注入或硬编码）
// ============================================================
const PRODUCT_INFO = {
    "IF": {"name": "沪深300", "multiplier": 300, "color": "#e74c3c"},
    "IH": {"name": "上证50", "multiplier": 300, "color": "#3498db"},
    "IC": {"name": "中证500", "multiplier": 200, "color": "#2ecc71"},
    "IM": {"name": "中证1000", "multiplier": 200, "color": "#9b59b6"},
};

// ============================================================
// App State
// ============================================================
let APP_DATA = null;
let currentProduct = '';
let currentMatrixMode = 'adj'; // 'adj' | 'raw' | 'raw_rate'
let tsChartInstance = null;

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    // 从 <script> 标签读取数据
    var appDataEl = document.getElementById('appData');
    var appConfigEl = document.getElementById('appConfig');
    
    if (appDataEl) {
        APP_DATA = JSON.parse(appDataEl.textContent);
    }
    if (appConfigEl) {
        var config = JSON.parse(appConfigEl.textContent);
        if (config.firstProduct) {
            currentProduct = config.firstProduct;
        }
    }
    
    if (!currentProduct && APP_DATA) {
        currentProduct = Object.keys(APP_DATA)[0] || '';
    }
    
    renderProduct(currentProduct);
});

// ============================================================
// Tab Switching
// ============================================================
function switchTab(btn, product) {
    currentProduct = product;
    document.querySelectorAll('.tab-btn').forEach(function(b) {
        b.classList.remove('active');
    });
    btn.classList.add('active');
    renderProduct(product);
}

function switchMatrix(mode) {
    currentMatrixMode = mode;
    document.querySelectorAll('.toggle-btn').forEach(function(b) {
        b.classList.remove('active');
    });
    var sel = '.toggle-btn[data-mode="' + mode + '"]';
    var targetBtn = document.querySelector(sel);
    if (targetBtn) {
        targetBtn.classList.add('active');
    }
    renderMatrix(currentProduct);
}

// ============================================================
// Main Render - 渲染指定品种的全部内容
// ============================================================
function renderProduct(product) {
    var d = APP_DATA[product];
    if (!d) return;

    var container = document.getElementById('mainContent');
    var latest = d.snapshot || {};
    var stats = d.statistics || [];
    var tsMap = d.time_series || {};

    // 更新时间戳
    var timeEl = document.getElementById('updateTime');
    if (latest.timestamp && timeEl) {
        var contractsStr = (latest.contracts || []).join(', ');
        timeEl.textContent = '最新数据: ' + latest.timestamp + '  |  活跃合约: ' + contractsStr;
    }

    // ---- 各模块 HTML ----
    var overviewHtml = buildContractOverview(latest);
    var matrixHtml = buildMatrixHtml(latest, stats);
    var statsHtml = buildStatsHtml(stats);
    var divHtml = buildDividendPanel(latest, stats);
    var chartHtml = buildChartPanel(product);

    // ---- 组装布局 ----
    container.innerHTML =
        overviewHtml +
        '<div class="grid-2">' +
        '<div>' + matrixHtml + divHtml + '</div>' +
        '<div>' + statsHtml + '</div></div>' +
        chartHtml;

    // 填充合约对下拉框
    fillPairSelector(tsMap, product);

    // 渲染图表
    renderChart(product);
}

// ============================================================
// Contract Overview Panel - 四合约总览（价格/基差/分红/调整后）
// 优先展示 C1/C2/C3/C4 四主力合约，其余折叠
// ============================================================
function buildContractOverview(latest) {
    var contracts = latest.contract_overview || [];
    var mainContracts = latest.main_contracts || [];
    var mainSet = {};
    for (var mi = 0; mi < mainContracts.length; mi++) {
        mainSet[mainContracts[mi]] = true;
    }
    // C1-C4 标签
    var contractLabels = ['C1\u5F53\u6708', 'C2\u4E0B\u6708', 'C3\u5F53\u5B63', 'C8\u8FDC\u5B63'];

    if (contracts.length === 0) return '';

    // 分离主力和非主力
    var mainList = [], otherList = [];
    for (var i = 0; i < contracts.length; i++) {
        if (mainSet[contracts[i].symbol]) {
            mainList.push(contracts[i]);
        } else {
            otherList.push(contracts[i]);
        }
    }
    // 按主力顺序排列
    mainList.sort(function(a, b) {
        return mainContracts.indexOf(a.symbol) - mainContracts.indexOf(b.symbol);
    });

    var html = '<div class="card" style="margin-bottom:16px;">' +
        '<div class="card-title"><span class="icon">\uD83D\uCC8C</span>合约总览' +
        '<span style="font-size:11px;font-weight:400;color:var(--text-muted);' +
        'margin-left:8px;">C1\u5F53\u6708 / C2\u4E0B\u6708 / C3\u5F53\u5B63 / C8\u8FDC\u5B63 \u00B7 \u5F53\u524D\u4EF7 / \u57FA\u5DEE / \u5206\u7EA2 / \u8C03\u6574\u540E</span></div>';

    // ---- 四主力合约卡片 ----
    html += '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;">';

    for (var i = 0; i < mainList.length; i++) {
        var c = mainList[i];
        var label = (i < contractLabels.length) ? contractLabels[i] : ('C' + (i+1));
        var adjColor = '#e6edf3';
        var adjLabel = '';
        if (c.adj_annualized < -0.5) { adjColor = 'var(--green)'; adjLabel = '(\u8D34\u6C34)'; }
        else if (c.adj_annualized > 0.5) { adjColor = 'var(--red)'; adjLabel = '(\u5347\u6C34)'; }
        else { adjColor = 'var(--text-muted)'; adjLabel = '(\u4E2D\u6027)'; }

        var divColor = c.dividend > 0.01 ? 'var(--yellow)' : 'var(--text-muted)';
        var divText = c.dividend.toFixed(1) + 'pt';
        if (c.dividend < 0.01) divText = '\u2014';

        html += '<div style="' +
            'flex:1;min-width:200px;background:#21262d;border-radius:8px;' +
            'padding:12px;border:1px solid var(--blue);position:relative;">';

        // 标签角标
        html += '<span style="position:absolute;top:-1px;left:8px;background:var(--blue);' +
            'color:#fff;font-size:10px;font-weight:700;padding:1px 8px;' +
            'border-radius:0 0 4px 4px;">' + label + '</span>';

        // 合约名 + 价格
        html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;margin-bottom:8px;">' +
            '<span style="font-size:16px;font-weight:700;color:var(--blue);">' +
            c.symbol + '</span>' +
            '<span style="font-size:20px;font-weight:700;font-family:Consolas,monospace;color:#fff;">' +
            c.price.toFixed(1) + '</span></div>';

        // 基差信息行
        html += '<div style="display:flex;flex-direction:column;gap:3px;font-size:12px;">' +
            '<div style="display:flex;justify-content:space-between;">' +
            '<span style="color:var(--text-muted);">\u539F\u59CB\u4EF7\u5DEE</span>' +
            '<span style="color:var(--text-muted);font-family:Consolas,monospace;">' +
            c.raw_spread.toFixed(1) + 'pt</span></div>' +
            '<div style="display:flex;justify-content:space-between;">' +
            '<span style="color:var(--text-muted);">\u539F\u59CB\u5E74\u5316</span>' +
            '<span style="color:var(--text-muted);font-family:Consolas,monospace;">' +
            c.raw_annualized.toFixed(2) + '%</span></div>' +
            '<div style="display:flex;justify-content:space-between;border-top:1px solid #30363d;padding-top:3px;">' +
            '<span style="color:var(--text-muted);\u5206\u7EA2\u5F71\u54CD</span>' +
            '<span style="color:' + divColor + ';font-family:Consolas,monospace;font-weight:600;">' +
            divText + '</span></div>' +
            '<div style="display:flex;justify-content:space-between;">' +
            '<span style="color:var(--text-muted);">\u8C03\u6574\u540E</span>' +
            '<span style="color:' + adjColor + ';font-family:Consolas,monospace;font-weight:700;">' +
            c.adj_annualized.toFixed(2) + '% ' + adjLabel + '</span></div>';

        html += '</div></div>'; // inner card close
    }
    html += '</div>'; // flex container close

    // ---- 其他合约（折叠显示） ----
    if (otherList.length > 0) {
        html += '<details><summary style="cursor:pointer;color:var(--text-muted);font-size:12px;' +
            'padding:4px 0;">' +
            '\u5176\u4ED6\u6D3B\u8DC3\u5408\u7EA6 (' + otherList.length + ')' +
            '</summary>';
        html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">';
        for (var j = 0; j < otherList.length; j++) {
            var oc = otherList[j];
            var oAdjColor = oc.adj_annualized < -0 ? 'var(--green)' :
                           (oc.adj_annualized > 0 ? 'var(--red)' : 'var(--text-muted)');
            html += '<div style="' +
                'flex:1;min-width:140px;background:#161b22;border-radius:6px;' +
                'padding:8px;border:1px solid var(--border);">';
            html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' +
                '<span style="font-size:13px;font-weight:600;color:var(--text-muted);">' +
                oc.symbol + '</span>' +
                '<span style="font-size:14px;font-weight:600;font-family:Consolas,monospace;color:#ccc;">' +
                oc.price.toFixed(1) + '</span></div>';
            html += '<div style="display:flex;justify-content:space-between;font-size:11px;">' +
                '<span style="color:var(--text-muted);">adj</span>' +
                '<span style="color:' + oAdjColor + ';font-family:Consolas,monospace;">' +
                oc.adj_annualized.toFixed(2) + '%</span></div>';
            html += '</div>';
        }
        html += '</div></details>';
    }

    html += '</div>'; // card close
    return html;
}

// ============================================================
// Matrix Panel - 价差矩阵热力图
// ============================================================
function buildMatrixHtml(latest, stats) {
    if (!latest.contracts || latest.contracts.length === 0) {
        return '<div class="card"><p style="color:var(--text-muted)">无矩阵数据</p></div>';
    }

    var contracts = latest.contracts;
    var mainContracts = latest.main_contracts || [];
    var mainSet = {};
    for (var mi = 0; mi < mainContracts.length; mi++) {
        mainSet[mainContracts[mi]] = true;
    }
    var n = contracts.length;

    // 构建信号映射: pair -> {zscore, signalText, signalClass}
    var signalMap = {};
    var hasSignal = false;
    if (stats && stats.length > 0) {
        for (var si = 0; si < stats.length; si++) {
            var s = stats[si];
            var zAbs = Math.abs(s.zscore);
            var sigClass = '', sigText = '';
            if (s.zscore > 1.5) {
                sigClass = 'signal-short'; sigText = '\u9AD8\u4F30';
                hasSignal = true;
            } else if (s.zscore < -1.5) {
                sigClass = 'signal-long'; sigText = '\u4F4E\u4F30';
                hasSignal = true;
            } else if (zAbs > 1.0) {
                sigClass = 'signal-weak';
                sigText = s.zscore > 0 ? '\u504F\u5F3A' : '\u504E\u5F31';
            }
            if (sigClass) {
                signalMap[s.pair] = {
                    zscore: s.zscore,
                    signalClass: sigClass,
                    signalText: sigText
                };
            }
        }
    }

    // 选择矩阵数据和格式化函数
    var matrix, fmt, titleSuffix, tooltipPrefix;
    if (currentMatrixMode === 'adj') {
        matrix = latest.matrix_adj;
        titleSuffix = '分红调整后年化%';
        tooltipPrefix = '年化基差率(%)';
        fmt = function(v) { return v === 0 ? '\u2014' : v.toFixed(2) + '%'; };
    } else if (currentMatrixMode === 'raw_rate') {
        matrix = latest.matrix_raw_rate;
        titleSuffix = '原始年化%';
        tooltipPrefix = '原始年化(%)';
        fmt = function(v) { return v === 0 ? '\u2014' : v.toFixed(2) + '%'; };
    } else {
        matrix = latest.matrix_raw;
        titleSuffix = '原始价差点数';
        tooltipPrefix = '价差(点)';
        fmt = function(v) { return v === 0 ? '\u2014' : v.toFixed(1); };
    }

    // 模式切换按钮
    var adjActive = currentMatrixMode === 'adj' ? ' active' : '';
    var rrActive = currentMatrixMode === 'raw_rate' ? ' active' : '';
    var rawActive = currentMatrixMode === 'raw' ? ' active' : '';

    var toggleBtns = '<div class="toggle-group">' +
        '<button class="toggle-btn' + adjActive +
        '" data-mode="adj" onclick="switchMatrix(\'adj\')">分红调整后 %</button>' +
        '<button class="toggle-btn' + rrActive +
        '" data-mode="raw_rate" onclick="switchMatrix(\'raw_rate\')">原始年化 %</button>' +
        '<button class="toggle-btn' + rawActive +
        '" data-mode="raw" onclick="switchMatrix(\'raw\')">原始价差 pt</button></div>';

    // 计算着色范围（排除对角线零值）
    var allVals = [];
    for (var ri = 0; ri < matrix.length; ri++) {
        for (var ci = 0; ci < matrix[ri].length; ci++) {
            if (matrix[ri][ci] !== 0) allVals.push(matrix[ri][ci]);
        }
    }
    var vmin = Math.min.apply(null, allVals);
    var vmax = Math.max.apply(null, allVals);

    // 构建 HTML
    var html = '<div class="card">' +
        '<div class="card-title"><span class="icon">\uD83D\uDCCA</span>价差矩阵（' +
        titleSuffix + '）<span style="font-size:11px;font-weight:400;' +
        'color:var(--text-muted);margin-left:8px;">行-列 = 近月-远月价差 · 蓝框=C1/C2/C3/C4</span></div>';
    html += toggleBtns;

    html += '<div class="matrix-wrapper"><table class="matrix">';
    html += '<tr><th style="width:80px;">合约</th>';
    for (var cidx = 0; cidx < contracts.length; cidx++) {
        var isMainHeader = mainSet[contracts[cidx]];
        var headerStyle = isMainHeader ?
            'class="matrix-label" style="color:var(--blue);font-weight:700;border-bottom:2px solid var(--blue);"' :
            '';
        html += '<th ' + headerStyle + '>' + contracts[cidx] + '</th>';
    }
    html += '</tr>';

    for (var i = 0; i < n; i++) {
        var isRowMain = mainSet[contracts[i]];
        var rowThStyle = isRowMain ?
            'class="matrix-label" style="color:var(--blue);font-weight:700;border-right:2px solid var(--blue);"' :
            'class="matrix-label"';
        html += '<tr><th ' + rowThStyle + '>' + contracts[i] + '</th>';
        for (var j = 0; j < n; j++) {
            var val = matrix[i][j];
            if (i === j) {
                html += '<td class="diagonal">\u2014</td>';
            } else {
                var color = heatColor(val, vmin, vmax);
                // 确定合约对名称 (行-列)
                var pairKey = contracts[i] + '/' + contracts[j];
                // 反向也查一下（因为矩阵存储可能反过来）
                var sig = signalMap[pairKey] || null;

                var titleStr = contracts[i] + '-' + contracts[j] + ': ' + tooltipPrefix + '=' + val.toFixed(2);
                if (sig) {
                    titleStr += ' | Z=' + (sig.zscore > 0 ? '+' : '') + sig.zscore.toFixed(2) + ' ' + sig.signalText;
                }

                // 单元格内容：数值 + 信号标记
                var cellContent = fmt(val);
                var extraStyle = '';
                var signalTag = '';

                // 主力合约交叉格加粗边框
                if (isRowMain && mainSet[contracts[j]]) {
                    extraStyle += 'border:2px solid rgba(88,166,255,0.5);';
                }

                if (sig && (sig.signalClass === 'signal-short' || sig.signalClass === 'signal-long')) {
                    // 有强信号时添加边框发光效果
                    if (sig.signalClass === 'signal-short') {
                        extraStyle = ';box-shadow:inset 0 0 0 2px rgba(248,81,73,0.6);';
                    } else {
                        extraStyle = ';box-shadow:inset 0 0 0 2px rgba(63,185,80,0.6);';
                    }
                    signalTag = '<span style="display:block;font-size:9px;margin-top:2px;opacity:0.9;">' +
                        sig.signalText + '</span>';
                } else if (sig && sig.signalClass === 'signal-weak') {
                    // 弱信号用下划线
                    extraStyle = ';text-decoration:underline;text-decoration-color:rgba(210,153,34,0.6);';
                }

                html += '<td style="background:' + color + extraStyle + '" title="' + titleStr + '">' +
                        cellContent + signalTag + '</td>';
            }
        }
        html += '</tr>';
    }
    html += '</table></div>';

    // Legend
    html += '<div class="legend">' +
        '<div class="legend-item"><div class="legend-color" style="background:#2ecc71"></div>' +
        '\u6DF1\u8D34\u6C34（远月便宜 → 多远空近机会）</div>' +
        '<div class="legend-item"><div class="legend-color" style="background:#fff;border:1px solid #555"></div>' +
        '\u4E2D\u6027\u533A\u95F4</div>' +
        '<div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div>' +
        '\u5347\u6C34/\u6D45\u8D34\u6C34（远月贵 → 多近空远机会）</div>' +
        '<div class="legend-item"><div class="legend-color" style="background:rgba(88,166,255,0.4);border:1px solid #58a6ff"></div>' +
        'C1/C2/C3/C4 \u56DB\u4E3B\u529B</div>' +
        '</div>';

    html += '</div>'; // card close
    return html;
}

// ============================================================
// Stats Panel - Z-Score偏离信号表
// ============================================================
function buildStatsHtml(stats) {
    var html = '<div class="card"><div class="card-title">' +
        '<span class="icon">\uD83D\uDCCF</span>Z-Score 偏离信号（按|Z|排序，Rolling ' +
        getLookback() + '根K线）</div>';

    if (stats.length === 0) {
        html += '<p style="color:var(--text-muted);padding:10px;">暂无数据</p>';
        html += '</div>';
        return html;
    }

    html += '<div style="overflow-x:auto;"><table style="width:100%;font-size:13px;">';
    // 表头
    html += '<tr style="color:var(--text-muted);font-size:11px;">' +
        '<th style="padding:6px;text-align:left;">合约对</th>' +
        '<th style="padding:6px;text-align:right;">当前值%</th>' +
        '<th style="padding:6px;text-align:right;">均值</th>' +
        '<th style="padding:6px;text-align:right;">\u03C3</th>' +
        '<th style="padding:6px;text-align:right;">Z-Score</th>' +
        '<th style="padding:6px;text-align:center;">信号</th>' +
        '<th style="padding:6px;text-align:right;">原始价差</th></tr>';

    for (var si = 0; si < stats.length; si++) {
        var s = stats[si];
        var zAbs = Math.abs(s.zscore);
        var signalClass = 'signal-neutral', signalText = '\u4E2D\u6027';

        if (s.zscore > 1.5) {
            signalClass = 'signal-short'; signalText = '\u9AD8\u4F30(\u7A7A\u8FDC\u591A\u8FD1)';
        } else if (s.zscore < -1.5) {
            signalClass = 'signal-long'; signalText = '\u4F4E\u4F30(\u591A\u8FDC\u7A7A\u8FD1)';
        } else if (zAbs > 1.0) {
            signalClass = '';
            signalText = s.zscore > 0 ? '\u504F\u5F3A' : '\u504E\u5F31';
        }

        // Z-Score 背景色
        var zBg = '#21262d';
        if (zAbs > 0.5) {
            var intensity = Math.min(0.3 + zAbs / 4, 0.8);
            if (s.zscore > 0) {
                zBg = 'rgba(248,81,73,' + intensity + ')';
            } else {
                zBg = 'rgba(63,185,80,' + intensity + ')';
            }
        }

        var zScoreStr = s.zscore > 0 ? '+' + s.zscore.toFixed(2) : s.zscore.toFixed(2);

        html += '<tr class="stats-row">' +
            '<td class="stats-pair">' + s.pair + '</td>' +
            '<td class="stats-val" style="text-align:right;">' + s.current + '%</td>' +
            '<td class="stats-val" style="text-align:right;">' + s.mean + '%</td>' +
            '<td class="stats-val" style="text-align:right;">' + s.std + '</td>' +
            '<td><span class="stats-z" style="background:' + zBg + ';color:#fff;">' +
            zScoreStr + '</span></td>' +
            '<td><span class="signal-badge ' + signalClass + '">' + signalText + '</span></td>' +
            '<td class="stats-val" style="text-align:right;">' + s.current_raw + 'pt</td></tr>';
    }

    html += '</table></div></div>';
    return html;
}

// ============================================================
// Price/Dividend Detail Panel - 原始价格、基差、分红影响
// ============================================================
function buildDividendPanel(latest) {
    var details = latest.pair_details || [];
    if (details.length === 0) return '';

    var html = '<div class="card"><div class="card-title">' +
        '<span class="icon">\uD83D\uDCCA</span>\u5408\u7EA6\u5BF9\u8BE6\u60C5' +
        '<span style="font-size:11px;font-weight:400;color:var(--text-muted);' +
        'margin-left:8px;">\u539F\u59CB\u4EF7\u683C / \u539F\u59CB\u57FA\u5DEE / \u5206\u7EA2\u8C03\u6574\u540E</span></div>';

    // 表格：每个合约对一行，显示完整信息
    html += '<div style="overflow-x:auto;"><table class="detail-table" ' +
        'style="width:100%;font-size:12px;border-collapse:collapse;">';

    // 表头
    html += '<tr style="color:var(--text-muted);font-size:11px;background:#21262d;">' +
        '<th style="padding:8px 10px;text-align:left;">\u5408\u7EA6\u5BF9</th>' +
        '<th style="padding:8px 6px;text-align:right;">\u8FD1\u6708</th>' +
        '<th style="padding:8px 6px;text-align:right;">\u8FDC\u6708</th>' +
        '<th style="padding:8px 6px;text-align:right;">\u539F\u59CB\u4EF7\u5DEE(pt)</th>' +
        '<th style="padding:8px 6px;text-align:right;">\u539F\u59CB\u5E74\u5316(%)</th>' +
        '<th style="padding:8px 6px;text-align:right;">\u5206\u7EA2(pt)</th>' +
        '<th style="padding:8px 6px;text-align:right;">\u8C03\u6574\u540E(%)</th>' +
        '</tr>';

    for (var pi = 0; pi < details.length; pi++) {
        var p = details[pi];

        // 分红颜色
        var divColor = '#8b949e';
        if (p.dividend > 0.01) divColor = 'var(--yellow)';
        else if (p.dividend < -0.01) divColor = 'var(--green)';

        // 调整后基差率颜色
        var adjClass = '';
        if (p.adj_annualized > 0.5) adjClass = 'color:var(--red);';
        else if (p.adj_annualized < -0.5) adjClass = 'color:var(--green);';

        html += '<tr style="border-bottom:1px solid #21262d;">' +
            '<td style="padding:7px 10px;font-weight:600;color:var(--blue);">' + p.pair + '</td>' +
            '<td style="padding:7px 6px;text-align:right;font-family:Consolas,monospace;">' + p.near_close + '</td>' +
            '<td style="padding:7px 6px;text-align:right;font-family:Consolas,monospace;">' + p.far_close + '</td>' +
            '<td style="padding:7px 6px;text-align:right;font-family:Consolas,monospace;">' + p.raw_spread.toFixed(1) + '</td>' +
            '<td style="padding:7px 6px;text-align:right;font-family:Consolas,monospace;color:var(--text-muted);">' + p.raw_annualized.toFixed(2) + '</td>' +
            '<td style="padding:7px 6px;text-align:right;font-weight:700;font-family:Consolas,monospace;' + divColor + '">' + p.dividend.toFixed(1) + '</td>' +
            '<td style="padding:7px 6px;text-align:right;font-weight:700;font-family:Consolas,monospace;' + adjClass + '">' + p.adj_annualized.toFixed(2) + '</td>' +
            '</tr>';
    }

    html += '</table></div>';
    
    // 底部说明
    html += '<div style="margin-top:8px;font-size:11px;color:var(--text-muted);line-height:1.6;">' +
        '<span style="display:inline-block;margin-right:16px;"><b>\u56FE\u4F8B:</b> \u5206\u7EA2=38pt</span>' +
        '\u5219 \u8C03\u6574\u540E\u57FA\u5DEE = \u539F\u59CB\u4EF7\u5DEE - (-38pt/\u671F\u9650\u5DEE) \u2248 \u539F\u59CB + \u5206\u7EA2\u8865\u503F' +
        '</div>';

    html += '</div>'; // card close
    return html;
}

// ============================================================
// Chart Panel
// ============================================================
function buildChartPanel(product) {
    return '<div class="card"><div class="card-title">' +
        '<span class="icon">\uD83D\uDCC8</span>\u5408\u7EA6\u5BF9\u5386\u53CB\u8D70\u52FF' +
        '<span style="font-size:11px;font-weight:400;color:var(--text-muted);' +
        'margin-left:8px;">\u539F\u59CB / \u5206\u7EA2\u8C03\u6574\u540E</span></div>' +
        '<div class="toggle-group" style="margin-bottom:10px;">' +
        '<select id="pairSelect" onchange="renderChart(\'' + product + '\')" ' +
        'style="background:var(--card-bg);color:var(--text);border:1px solid var(--border);' +
        'border-radius:6px;padding:6px 10px;font-size:13px;width:220px;"></select>' +
        '</div><div class="chart-container"><canvas id="tsChart"></canvas></div></div>';
}

function fillPairSelector(tsMap, product) {
    var selectEl = document.getElementById('pairSelect');
    if (!selectEl) return;

    var pairs = Object.keys(tsMap).map(function(k) { return k.replace('|', '/'); });
    for (var i = 0; i < pairs.length; i++) {
        var opt = document.createElement('option');
        opt.value = pairs[i];
        opt.text = pairs[i];
        selectEl.appendChild(opt);
    }
}

// ============================================================
// Chart.js - Time Series Rendering
// ============================================================
function renderChart(product) {
    var d = APP_DATA[product];
    if (!d) return;

    var tsMap = d.time_series || {};
    var selectEl = document.getElementById('pairSelect');
    var selectedPair = selectEl ? selectEl.value : null;
    if (!selectedPair) return;

    var parts = selectedPair.split('/');
    var near = parts[0], far = parts[1];
    var key = near + '|' + far;
    var series = tsMap[key];
    if (!series || series.length === 0) return;

    var labels = series.map(function(s) { return s.time.substring(5, 16); }); // MM-DD HH:MM
    var rawData = series.map(function(s) { return s.raw; });
    var rawRateData = series.map(function(s) { return s.raw_rate; });
    var adjData = series.map(function(s) { return s.adj_rate; });

    var ctx = document.getElementById('tsChart');
    if (!ctx) return;
    if (tsChartInstance) tsChartInstance.destroy();

    var prodColor = '#58a6ff';
    if (PRODUCT_INFO[product] && PRODUCT_INFO[product].color) {
        prodColor = PRODUCT_INFO[product].color;
    }

    tsChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '\u539F\u59CB\u4EF7\u5DEE(pt)',
                    data: rawData,
                    borderColor: '#6e7681',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y_raw',
                    tension: 0.2,
                },
                {
                    label: '\u539F\u59CB\u5E74\u5316(%)',
                    data: rawRateData,
                    borderColor: '#f0883e',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y_pct',
                    borderDash: [5, 3],
                    tension: 0.2,
                },
                {
                    label: '\u5206\u7EA2\u8C03\u6574\u540E(%)',
                    data: adjData,
                    borderColor: prodColor,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y_pct',
                    tension: 0.3,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { 
                    labels: { color: '#8b949e', font: { size: 10 }, boxWidth: 12 } 
                },
                tooltip: {
                    callbacks: {
                        title: function(items) {
                            return items[0].label;
                        },
                        afterBody: function(items) {
                            var idx = items[0].dataIndex;
                            var s = series[idx];
                            return [
                                '',
                                '\u8FD1\u6708: ' + s.near_price + '  |  \u8FDC\u6708: ' + s.far_price,
                                '\u5206\u7EA2\u9884\u6D4B: ' + s.dividend + 'pt'
                            ];
                        }
                    }
                },
            },
            scales: {
                x: {
                    ticks: { color: '#8b949e', font: { size: 10 }, maxTicksLimit: 20 },
                    grid: { color: 'rgba(48,54,61,0.4)' }
                },
                y_raw: {
                    position: 'left',
                    title: { display: true, text: '\u4EF7\u5DEE(\u70B9)', color: '#6e7681' },
                    ticks: { color: '#6e7681', font: { size: 10 } },
                    grid: { color: 'rgba(48,54,61,0.4)' }
                },
                y_pct: {
                    position: 'right',
                    title: { display: true, text: '\u5E74\u5316\u57FA\u5DEE\u7387(%)', color: prodColor },
                    ticks: { color: prodColor, font: { size: 10 } },
                    grid: { drawOnChart: false }
                },
            }
        }
    });
}

// ============================================================
// Helpers
// ============================================================
function heatColor(val, vmin, vmax) {
    if (vmax === vmin) return 'rgba(128,128,128,0.85)';
    var t = (val - vmin) / (vmax - vmin);
    var r, g, b;
    if (t < 0.5) {
        r = Math.round(255 * t * 2);
        g = Math.round(180 + 75 * t * 2);
        b = Math.round(150 + 105 * t * 2);
    } else {
        r = 255;
        g = Math.round(255 - 175 * (t - 0.5) * 2);
        b = Math.round(255 - 205 * (t - 0.5) * 2);
    }
    return 'rgba(' + r + ',' + g + ',' + b + ',0.85)';
}

function getLookback() {
    return 288;
}
