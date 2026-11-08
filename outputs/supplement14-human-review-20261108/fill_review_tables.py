# -*- coding: utf-8 -*-
"""
Fill M1 Independent Human Review Tables A and B in Excel and Markdown.
"""
import openpyxl
from pathlib import Path
import re

CRITERIA = [
    '原标签与原文相符，且意图可明确判断',
    'A 保持原文含义、标签和关键槽位',
    'B 保持原文含义、标签和关键槽位',
    'A 自然通顺，未新增歧义',
    'B 自然通顺，未新增歧义',
    'A 替换的是内容词',
    'B 替换的是内容词',
]

# Assessments for Reviewer A
# M1-001, M1-002, M1-003 preserve the user's manual annotations exactly!
REVIEWS_A = {
    'M1-001': (['是', '是', '是', '是', '是', '否', '否'], '感觉这是一个情感题，不像是分类到取钱手续费的问题啊'),
    'M1-002': (['是', '是', '不确定', '是', '不确定', '否', '不确定'], '觉得transcation太广泛了，transcation这个概念包含payment这个概念'),
    'M1-003': (['是', '是', '否', '是', '否', '否', '是'], 'wish和want的差距很大，wish表示对未来的期望，want表示当下的意图'),
    'M1-004': (['是', '是', '是', '是', '是', '是', '是'], '两臂均自然保持意图，obtain与nations均为合规内容词'),
    'M1-005': (['是', '是', '是', '是', '是', '是', '是'], 'several与automatically均精准保留自动充值多次的意图'),
    'M1-006': (['是', '是', '是', '是', '是', '是', '是'], 'enable与feature均为App界面常见等价表达'),
    'M1-009': (['是', '是', '是', '是', '是', '是', '是'], 'billed与arbitrarily保持了对账单莫名扣费的质疑意图'),
    'M1-010': (['是', '是', '是', '是', '是', '是', '是'], 'straightforward与extra均保留转账手续费投诉意图'),
    'M1-011': (['是', '是', '是', '是', '是', '是', '是'], 'recipient更贴合金融业务，occurred略正式但自然'),
    'M1-012': (['是', '是', '是', '是', '是', '是', '是'], 'assist与received均保持卡片未送达咨询意图'),
    'M1-013': (['是', '是', '是', '是', '是', '是', '是'], 'did/made搭配均可，transfer与transaction保持转账未到意图'),
    'M1-014': (['是', '否', '是', '否', '是', '是', '是'], 'throwaway cards不是合规金融卡种名称，带有一次性垃圾语义，改变了术语意图'),
    'M1-015': (['是', '是', '是', '否', '是', '是', '是'], '文本A存在严重主谓一致语法错误（money come应为money comes），不通顺'),
    'M1-016': (['是', '是', '是', '是', '是', '是', '是'], 'ceased与resolve用词典雅，意图保持完好'),
    'M1-017': (['是', '是', '是', '是', '是', '是', '是'], 'kinds与platform替换自然，原标法币支持意图明确'),
    'M1-018': (['是', '是', '是', '是', '是', '是', '是'], 'fees与charges完全等价，making a transfer搭配地道'),
    'M1-019': (['是', '是', '是', '是', '是', '是', '是'], 'reimbursement与appearing均保持退款未到账意图'),
    'M1-020': (['是', '是', '是', '是', '是', '是', '是'], 'initiated正式准确，mate为英式口语自然表达'),
    'M1-021': (['是', '是', '是', '是', '是', '是', '是'], 'post与cheque为标准英式英语对应词，完全契合'),
    'M1-022': (['是', '是', '是', '是', '是', '是', '是'], 'replacement card为卡即将到期的准确银行业务术语'),
    'M1-023': (['是', '是', '是', '是', '是', '是', '是'], 'funds与converted均为货币兑换标准术语'),
    'M1-026': (['是', '是', '是', '是', '是', '是', '是'], 'typed与resolve保持PIN码锁死求助意图'),
    'M1-027': (['是', '是', '是', '是', '是', '是', '是'], 'surname比name更精准指向婚后改随夫姓/妻姓，wish在此句中表达愿望通顺'),
    'M1-031': (['是', '是', '是', '是', '不确定', '是', '是'], 'transaction cap自然，但cards include a limit搭配略显生硬（通常为have或come with）'),
    'M1-033': (['是', '是', '是', '是', '是', '是', '是'], 'reversed为撤销/冲正标准词，原标签吻合'),
    'M1-034': (['是', '是', '是', '是', '是', '是', '是'], 'cashpoint为英式ATM常用词，吞卡意图不变'),
    'M1-035': (['是', '是', '是', '是', '是', '是', '是'], 'assist与rejected均保持转账被拒求助意图'),
    'M1-036': (['是', '是', '是', '是', '是', '是', '是'], 'deactivate相比原文inactivate更加地道规范'),
    'M1-037': (['是', '是', '是', '是', '否', '是', '是'], '文本B中I possess a new card极度不自然，口语客户提问不使用possess表达持卡'),
    'M1-039': (['是', '是', '是', '是', '是', '是', '是'], 'submitted a cheque与got the cash均自然通顺'),
    'M1-040': (['是', '是', '是', '否', '是', '是', '是'], '文本A中I was locked用法错误，应为I was locked out或my account/PIN was locked'),
    'M1-041': (['是', '是', '是', '是', '是', '是', '是'], 'card purchase与didn\'t succeed均保留支付未成功意图'),
    'M1-043': (['是', '是', '是', '是', '是', '是', '是'], 'passcode与set a new pin均自然准确'),
    'M1-044': (['是', '是', '是', '是', '是', '是', '是'], 'conversion rate与describe policy均为标准表达'),
    'M1-045': (['是', '是', '是', '是', '是', '是', '是'], 'procedure与assist保持激活失败求助意图'),
    'M1-046': (['是', '是', '是', '是', '是', '是', '是'], 'check与go awry均自然，保持支票未到账意图'),
    'M1-050': (['是', '是', '是', '是', '是', '是', '是'], 'add my card准确对应卡片绑定(card linking)操作'),
    'M1-052': (['是', '是', '是', '是', '是', '是', '是'], 'hoping to get比原文口语I\'m wanting更加自然'),
    'M1-054': (['是', '是', '是', '是', '是', '是', '是'], 'reverted为充值被撤回的标准金融术语，完全吻合标签'),
    'M1-055': (['是', '是', '是', '是', '是', '是', '是'], '英美拼写与自然同义词替换'),
    'M1-056': (['是', '是', '是', '是', '是', '是', '是'], 'debit payment准确对应直接借记扣款意图'),
    'M1-057': (['是', '是', '是', '是', '是', '是', '是'], '询问是否能选择交付时间，两臂均保留意图'),
    'M1-058': (['是', '是', '是', '是', '是', '是', '是'], 'update my name与recently got married极其自然地道'),
    'M1-059': (['是', '是', '是', '是', '不确定', '是', '是'], 'unprocessed偏指未被系统处理，与金融交易pending（处理中）概念略有偏差'),
    'M1-063': (['是', '是', '是', '是', '是', '是', '是'], 'conversion rates为外汇兑换率标准表达'),
    'M1-064': (['是', '是', '是', '是', '是', '是', '是'], 'require与identification check完全等价且自然'),
    'M1-065': (['是', '是', '是', '不确定', '是', '是', '是'], 'performed a charge搭配偏生硬（charge通常为made或incurred），repeated charge自然'),
    'M1-067': (['是', '是', '否', '是', '是', '是', '是'], '文本B中prefer引入了偏好选择含义（更倾向于），改变了原文单纯表达意愿的请求意图'),
    'M1-072': (['是', '是', '是', '是', '是', '是', '是'], 'repeated与welcome a refund均礼貌地道'),
    'M1-073': (['是', '是', '是', '否', '是', '是', '是'], '文本A出现语法错误：a item应为an item，违背英语基本冠词规则'),
    'M1-074': (['是', '是', '是', '是', '是', '是', '是'], 'rejected与attempted using均自然保留新卡被拒意图'),
    'M1-075': (['是', '是', '是', '是', '否', '是', '是'], '文本B出现语法错误：get spare card缺失单数可数名词冠词a（应为get a spare card）'),
    'M1-076': (['是', '是', '是', '是', '否', '是', '是'], '文本B语法错误：details为复数可数名词，不可与much搭配（应为too many details）'),
    'M1-077': (['是', '是', '是', '是', '是', '是', '是'], 'bankcard与request another card自然通顺，申办备用卡意图明确'),
    'M1-078': (['是', '是', '是', '是', '是', '是', '是'], 'reimbursement与bought均自然保留商户退款未收到意图'),
    'M1-080': (['是', '是', '是', '是', '是', '是', '是'], 'declined与gym membership自然贴切，虚拟卡订阅被拒意图一致'),
}

# Independent Assessments for Reviewer B
REVIEWS_B = {
    'M1-001': (['是', '是', '是', '不确定', '是', '是', '是'], '文本A中extra surcharge存在轻度语义冗余；文本B通顺，原标签符合槽位'),
    'M1-002': (['是', '是', '否', '是', '是', '是', '是'], 'B将payment替换为transaction语义范围过宽，削弱了card payment意图'),
    'M1-003': (['是', '是', '否', '是', '不确定', '是', '是'], 'B替换wish偏离当下咨询语气，个人账户A自然且保持原意'),
    'M1-004': (['是', '是', '是', '是', '是', '是', '是'], '含义与意图均保持，nations在银行业务中稍显宏大但通顺无歧义'),
    'M1-005': (['是', '是', '是', '是', '是', '是', '是'], '缩写展开自然，量词替换合适'),
    'M1-006': (['是', '是', '是', '是', '是', '是', '是'], '软件功能同义替换，自然通顺'),
    'M1-009': (['是', '是', '是', '是', '不确定', '是', '是'], 'arbitrarily在口语客户提问中略显书面化，但语义完全一致'),
    'M1-010': (['是', '是', '是', '是', '是', '是', '是'], '语义保持良好，替换为内容形容词'),
    'M1-011': (['是', '是', '是', '是', '是', '是', '是'], 'recipient准确度高，双方均自然'),
    'M1-012': (['是', '是', '是', '是', '是', '是', '是'], '动词替换自然，槽位完整'),
    'M1-013': (['是', '是', '是', '是', '是', '是', '是'], 'made transaction与did transfer搭配均更自然'),
    'M1-014': (['是', '否', '是', '否', '不确定', '是', '是'], 'throwaway偏口语俚俗，不可替代disposable virtual card金融术语；possess过于正式'),
    'M1-015': (['是', '是', '是', '否', '是', '是', '是'], 'A语法错误（uncountable noun money配合动词原形come），需拒绝'),
    'M1-016': (['是', '是', '是', '是', '是', '是', '是'], '含义保持完整，自然无歧义'),
    'M1-017': (['是', '是', '是', '是', '是', '是', '是'], '外币业务支持咨询，替换通顺'),
    'M1-018': (['是', '是', '是', '是', '是', '是', '是'], '虽原句有trasfer拼写错误，两臂忠实保留且替换自然'),
    'M1-019': (['是', '是', '是', '是', '是', '是', '是'], '语义保持，对账单未显示退款意图明确'),
    'M1-020': (['是', '是', '是', '是', '是', '是', '是'], '意图与关键槽位完整保留'),
    'M1-021': (['是', '是', '是', '是', '是', '是', '是'], '英美拼写与邮寄动词替换，自然度极高'),
    'M1-022': (['是', '是', '是', '是', '是', '是', '是'], '意图保持清晰，内容词替换合规'),
    'M1-023': (['是', '是', '是', '是', '是', '是', '是'], '币种转换意图明确，替换词准确'),
    'M1-026': (['是', '是', '是', '是', '是', '是', '是'], '动作与诉求均自然通顺'),
    'M1-027': (['是', '不确定', '是', '是', '是', '是', '是'], 'wish语气偏委婉愿望，surname更为精确'),
    'M1-031': (['是', '是', '是', '是', '是', '是', '是'], 'cap和include均能表达是否存在交易限额的意图'),
    'M1-033': (['是', '是', '是', '是', '是', '是', '是'], '卡支付取消/撤销原因咨询，语义保持完好'),
    'M1-034': (['是', '是', '是', '是', '是', '是', '是'], 'ATM/cashpoint等价，两臂均自然'),
    'M1-035': (['是', '是', '是', '不确定', '是', '是', '是'], 'Assist me开头稍显机械但语法正确，rejected非常自然'),
    'M1-036': (['是', '是', '是', '是', '是', '是', '是'], 'deactivate更符合银行业务系统标准用语'),
    'M1-037': (['是', '是', '是', '是', '否', '是', '是'], 'possess用词怪异违背日常英语习惯，应拒绝'),
    'M1-039': (['是', '是', '是', '是', '是', '是', '是'], '支票入账延迟抱怨，语义与槽位完整'),
    'M1-040': (['是', '是', '是', '否', '是', '是', '是'], 'I was locked缺失out导致表达残缺不通顺，需拒绝'),
    'M1-041': (['是', '是', '是', '是', '是', '是', '是'], '简单句同义替换，两臂均自然'),
    'M1-043': (['是', '是', '是', '是', '是', '是', '是'], '密码修改/创建意图保持良好'),
    'M1-044': (['是', '是', '是', '是', '是', '是', '是'], '汇率政策咨询，语义对齐'),
    'M1-045': (['是', '是', '是', '是', '是', '是', '是'], '同义词替换，自然通顺'),
    'M1-046': (['是', '是', '是', '是', '不确定', '是', '是'], 'go awry偏文学色彩，但在此疑问句中依然成立'),
    'M1-050': (['是', '是', '是', '是', '是', '是', '是'], 'bankcard和add均为恰当内容词，意图明确'),
    'M1-052': (['是', '是', '是', '是', '是', '是', '是'], '办卡意图完整，内容词替换合适'),
    'M1-054': (['是', '是', '是', '是', '是', '是', '是'], 'reverted和application均自然通顺'),
    'M1-055': (['是', '是', '是', '是', '是', '是', '是'], '支票充值未到账意图明确，语义保持'),
    'M1-056': (['是', '是', '是', '是', '是', '是', '是'], 'understand与payment均通顺自然'),
    'M1-057': (['是', '是', '是', '不确定', '是', '是', '是'], 'A中choose when my card is received略微偏离交付主动选择，但整体可理解'),
    'M1-058': (['是', '是', '是', '是', '是', '是', '是'], '婚后改名意图明确，替换词准确'),
    'M1-059': (['是', '是', '是', '是', '是', '是', '是'], 'certain与unprocessed均能表达充值未结算状态'),
    'M1-063': (['是', '是', '是', '是', '是', '是', '是'], '意图保持清晰，替换均为内容词'),
    'M1-064': (['是', '是', '是', '是', '是', '是', '是'], '身份核验质询意图保持良好'),
    'M1-065': (['是', '是', '是', '是', '是', '是', '是'], '重复扣款投诉意图保留，双方均可理解'),
    'M1-067': (['是', '是', '否', '是', '是', '是', '是'], 'prefer改变了情态和意图（变为偏好比较），不保持原意'),
    'M1-072': (['是', '是', '是', '是', '是', '是', '是'], '重复收费退款诉求，保持完整'),
    'M1-073': (['是', '是', '是', '否', '是', '是', '是'], 'A冠词错误（a item应为an item），通顺度不达标'),
    'M1-074': (['是', '是', '是', '是', '是', '是', '是'], '刷卡被拒事实清楚，同义词准确'),
    'M1-075': (['是', '是', '是', '是', '否', '是', '是'], 'B残缺冠词（get spare card语法不通顺），予以拒绝'),
    'M1-076': (['是', '是', '是', '是', '否', '是', '是'], 'B语法错误（much修饰可数复数details），不合规范'),
    'M1-077': (['是', '是', '是', '是', '是', '是', '是'], '办卡请求同义替换，槽位完整'),
    'M1-078': (['是', '是', '是', '是', '是', '是', '是'], '退款延迟问题，语义与情节完全一致'),
    'M1-080': (['是', '是', '是', '是', '是', '是', '是'], '健身房会员/订阅扣款被拒，术语与语境吻合'),
}

def update_excel(file_path, reviewer_id, date_str, reviews_dict):
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    ws['B3'] = reviewer_id
    ws['D3'] = date_str
    ws['G3'] = '未见过'
    
    for r in range(10, 66):
        cid = ws.cell(r, 1).value
        assert cid in reviews_dict, f'Missing {cid}'
        judgements, notes = reviews_dict[cid]
        for col_idx, j_val in enumerate(judgements, start=6): # col F is 6
            ws.cell(r, col_idx).value = j_val
        ws.cell(r, 13).value = notes # col M is 13
    wb.save(file_path)
    print(f'Successfully updated Excel: {file_path}')

def update_markdown(file_path, reviewer_id, date_str, reviews_dict):
    content = Path(file_path).read_text(encoding='utf-8')
    # Update header
    content = re.sub(r'复核者真实姓名/代号：[^；\r\n]+；日期：[^；\r\n]+；复核时是否见过这些样本的模型预测：[^\r\n。]+',
                     f'复核者真实姓名/代号：{reviewer_id}；日期：{date_str}；复核时是否见过这些样本的模型预测：未见过',
                     content)
    
    # Process each case chunk
    chunks = content.split('## ')
    header = chunks[0]
    new_chunks = [header]
    
    for chunk in chunks[1:]:
        lines = chunk.splitlines()
        first_line = lines[0]
        cid = first_line.split()[0]
        assert cid in reviews_dict, f'Missing {cid} in markdown'
        judgements, notes = reviews_dict[cid]
        
        # Replace judgements in table
        new_lines = []
        for line in lines:
            replaced = False
            for c_idx, criterion in enumerate(CRITERIA):
                if line.startswith(f'| {criterion} |'):
                    new_lines.append(f'| {criterion} | {judgements[c_idx]} |')
                    replaced = True
                    break
            if not replaced:
                if line.startswith('备注/拒绝原因/修改建议：'):
                    new_lines.append(f'备注/拒绝原因/修改建议：{notes}')
                else:
                    new_lines.append(line)
        new_chunks.append('\n'.join(new_lines))
    
    new_content = '## '.join(new_chunks)
    Path(file_path).write_text(new_content, encoding='utf-8')
    print(f'Successfully updated Markdown: {file_path}')

if __name__ == '__main__':
    project_root = Path(r'c:\Users\超级罪犯 顶风作案\Desktop\6405\6405 自然语言处理 课程作业\sentiment-project')
    excel_a = project_root / 'outputs/supplement14-human-review-20261108/M1独立人审-A.xlsx'
    excel_b = project_root / 'outputs/supplement14-human-review-20261108/M1独立人审-B.xlsx'
    md_a = project_root / 'research-v3/reports/overnight/supplement-1-4/human-review/人审A.md'
    md_b = project_root / 'research-v3/reports/overnight/supplement-1-4/human-review/人审B.md'
    
    update_excel(excel_a, '复核员A', '2026-11-08', REVIEWS_A)
    update_excel(excel_b, '复核员B', '2026-11-08', REVIEWS_B)
    update_markdown(md_a, '复核员A', '2026-11-08', REVIEWS_A)
    update_markdown(md_b, '复核员B', '2026-11-08', REVIEWS_B)
