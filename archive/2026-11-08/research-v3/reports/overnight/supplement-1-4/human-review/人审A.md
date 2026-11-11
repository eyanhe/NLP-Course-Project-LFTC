# M1 独立人审表 A

共 56 对 AI 编辑草稿，来自原预注册候选顺序的前80条。实际完成真人复核：0。目标为50个双人同意的有效对；当前不保证达到。

复核者真实姓名/代号：复核员A；日期：2026-11-08；复核时是否见过这些样本的模型预测：未见过。

请独立填写本文件，不看另一位复核者的表、模型预测或 arm-key。A/B文本顺序按样本固定打乱，未标出哪一臂是线索干预。标签是数据集原标签，也需要你核对。

每项请将“待填”改为“是 / 否 / 不确定”。未填写和不确定均不视为通过。两种编辑都应保持原文含义和原标签，不改变动作、对象、否定、数额、币种及其他关键槽位；两臂都应自然，替换应是内容词。

如果需要修改文本，请在备注给出建议，不直接改表中的原始草稿。修改后须重新确认两臂，先冻结最终文本再进行模型预测。完成后把此文件发回即可。

本文本中的大小写、拼写和原始语法问题保留；它们也可能令候选不适合实验，请据实拒绝。

## M1-001 · banking77/train/9398

原标签：cash_withdrawal_charge

原文：I hate this ATM, it charged me an extra fee, Why did it charge?

文本 A：I hate this ATM, it charged me an extra surcharge, Why did it charge?

文本 B：I dislike this ATM, it charged me an extra fee, Why did it charge?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 否 |
| B 替换的是内容词 | 否 |

备注/拒绝原因/修改建议：感觉这是一个情感题，不像是分类到取钱手续费的问题啊
## M1-002 · banking77/train/3484

原标签：card_payment_not_recognised

原文：There's a payment with my card that I definitely didn't do myself, never seen that name before.

文本 A：There's a payment with my bankcard that I definitely didn't do myself, never seen that name before.

文本 B：There's a transaction with my card that I definitely didn't do myself, never seen that name before.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 不确定 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 不确定 |
| A 替换的是内容词 | 否 |
| B 替换的是内容词 | 不确定 |

备注/拒绝原因/修改建议：觉得transcation太广泛了，transcation这个概念包含payment这个概念
## M1-003 · banking77/train/1578

原标签：age_limit

原文：If I want my son to have his own account is that possible?

文本 A：If I want my son to have his personal account is that possible?

文本 B：If I wish my son to have his own account is that possible?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 否 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 否 |
| A 替换的是内容词 | 否 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：wish和want的差距很大，wish表示对未来的期望，want表示当下的意图
## M1-004 · banking77/train/9965

原标签：country_support

原文：In which countries can I get a card?

文本 A：In which countries can I obtain a card?

文本 B：In which nations can I get a card?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：两臂均自然保持意图，obtain与nations均为合规内容词
## M1-005 · banking77/train/1204

原标签：automatic_top_up

原文：Can I auto top up multiple times?

文本 A：Can I auto top up several times?

文本 B：Can I automatically top up multiple times?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：several与automatically均精准保留自动充值多次的意图
## M1-006 · banking77/train/1193

原标签：automatic_top_up

原文：I'm trying to activate the auto-top up function. Where is it?

文本 A：I'm trying to enable the auto-top up function. Where is it?

文本 B：I'm trying to activate the auto-top up feature. Where is it?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：enable与feature均为App界面常见等价表达
## M1-009 · banking77/train/655

原标签：extra_charge_on_statement

原文：Why did I get charged one pound seemingly randomly?

文本 A：Why did I get billed one pound seemingly randomly?

文本 B：Why did I get charged one pound seemingly arbitrarily?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：billed与arbitrarily保持了对账单莫名扣费的质疑意图
## M1-010 · banking77/train/7241

原标签：transfer_fee_charged

原文：I do not like this, I made a simple transfer but I have an additional charge for doing so.

文本 A：I do not like this, I made a straightforward transfer but I have an additional charge for doing so.

文本 B：I do not like this, I made a simple transfer but I have an extra charge for doing so.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：straightforward与extra均保留转账手续费投诉意图
## M1-011 · banking77/train/2855

原标签：transfer_not_received_by_recipient

原文：I just sent money but the person I sent it to doesn't see it. What happened?

文本 A：I just sent money but the person I sent it to doesn't see it. What occurred?

文本 B：I just sent money but the recipient I sent it to doesn't see it. What happened?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：recipient更贴合金融业务，occurred略正式但自然
## M1-012 · banking77/train/70

原标签：card_arrival

原文：My card has not been delivered yet and its been a week since I ordered it, please help

文本 A：My card has not been delivered yet and its been a week since I ordered it, please assist

文本 B：My card has not been received yet and its been a week since I ordered it, please help

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：assist与received均保持卡片未送达咨询意图
## M1-013 · banking77/train/2849

原标签：transfer_not_received_by_recipient

原文：I just did a money transaction but its not showing they received it

文本 A：I just did a money transfer but its not showing they received it

文本 B：I just made a money transaction but its not showing they received it

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：did/made搭配均可，transfer与transaction保持转账未到意图
## M1-014 · banking77/train/4418

原标签：disposable_card_limits

原文：How many disposable cards can you own?

文本 A：How many throwaway cards can you own?

文本 B：How many disposable cards can you possess?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 否 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 否 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：throwaway cards不是合规金融卡种名称，带有一次性垃圾语义，改变了术语意图
## M1-015 · banking77/train/6532

原标签：verify_source_of_funds

原文：Are there steps to see where my funds come from?

文本 A：Are there steps to see where my money come from?

文本 B：Are there procedures to see where my funds come from?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 否 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本A存在严重主谓一致语法错误（money come应为money comes），不通顺
## M1-016 · banking77/train/1797

原标签：contactless_not_working

原文：My contactless stopped working today for no good reason. It's not working no matter where I try to use it. Can you please help me fix it?

文本 A：My contactless ceased working today for no good reason. It's not working no matter where I try to use it. Can you please help me fix it?

文本 B：My contactless stopped working today for no good reason. It's not working no matter where I try to use it. Can you please help me resolve it?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：ceased与resolve用词典雅，意图保持完好
## M1-017 · banking77/train/912

原标签：fiat_currency_support

原文：What types of foreign money can be used with this service.?

文本 A：What kinds of foreign money can be used with this service.?

文本 B：What types of foreign money can be used with this platform.?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：kinds与platform替换自然，原标法币支持意图明确
## M1-018 · banking77/train/1856

原标签：top_up_by_bank_transfer_charge

原文：If I top-up by using a trasfer are there charges that applied?

文本 A：If I top-up by using a trasfer are there fees that applied?

文本 B：If I top-up by making a trasfer are there charges that applied?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：fees与charges完全等价，making a transfer搭配地道
## M1-019 · banking77/train/5742

原标签：Refund_not_showing_up

原文：No refund showing in my statement.

文本 A：No reimbursement showing in my statement.

文本 B：No refund appearing in my statement.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：reimbursement与appearing均保持退款未到账意图
## M1-020 · banking77/train/2826

原标签：transfer_not_received_by_recipient

原文：I made a transfer to my friend earlier and she hasn't received it yet. What's the hold up?

文本 A：I initiated a transfer to my friend earlier and she hasn't received it yet. What's the hold up?

文本 B：I made a transfer to my mate earlier and she hasn't received it yet. What's the hold up?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：initiated正式准确，mate为英式口语自然表达
## M1-021 · banking77/train/7948

原标签：top_up_by_cash_or_cheque

原文：Can I mail in a check to top up?

文本 A：Can I post in a check to top up?

文本 B：Can I mail in a cheque to top up?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：post与cheque为标准英式英语对应词，完全契合
## M1-022 · banking77/train/9554

原标签：card_about_to_expire

原文：I need a new card before it expires.

文本 A：I need a replacement card before it expires.

文本 B：I require a new card before it expires.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：replacement card为卡即将到期的准确银行业务术语
## M1-023 · banking77/train/1364

原标签：exchange_via_app

原文：Can money be exchanged from one currency to another?

文本 A：Can funds be exchanged from one currency to another?

文本 B：Can money be converted from one currency to another?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：funds与converted均为货币兑换标准术语
## M1-026 · banking77/train/1683

原标签：pin_blocked

原文：I entered a wrong PIN too many times. How can I fix this?

文本 A：I entered a wrong PIN too many times. How can I resolve this?

文本 B：I typed a wrong PIN too many times. How can I fix this?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：typed与resolve保持PIN码锁死求助意图
## M1-027 · banking77/train/3682

原标签：edit_personal_details

原文：I just had a wedding and I want to take my spouses name.

文本 A：I just had a wedding and I wish to take my spouses name.

文本 B：I just had a wedding and I want to take my spouses surname.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：surname比name更精准指向婚后改随夫姓/妻姓，wish在此句中表达愿望通顺
## M1-031 · banking77/train/4344

原标签：disposable_card_limits

原文：Do the disposable cards have a transaction limit?

文本 A：Do the disposable cards have a transaction cap?

文本 B：Do the disposable cards include a transaction limit?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 不确定 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：transaction cap自然，但cards include a limit搭配略显生硬（通常为have或come with）
## M1-033 · banking77/train/6766

原标签：reverted_card_payment?

原文：What reasons would cause my card payment to be cancelled?

文本 A：What reasons would cause my card payment to be reversed?

文本 B：What factors would cause my card payment to be cancelled?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：reversed为撤销/冲正标准词，原标签吻合
## M1-034 · banking77/train/6254

原标签：card_swallowed

原文：I didn't get my card back from the ATM.

文本 A：I didn't receive my card back from the ATM.

文本 B：I didn't get my card back from the cashpoint.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：cashpoint为英式ATM常用词，吞卡意图不变
## M1-035 · banking77/train/5661

原标签：declined_transfer

原文：Help me. My transfer was declined. How could this happen?

文本 A：Assist me. My transfer was declined. How could this happen?

文本 B：Help me. My transfer was rejected. How could this happen?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：assist与rejected均保持转账被拒求助意图
## M1-036 · banking77/train/6206

原标签：terminate_account

原文：I want to inactivate my account.

文本 A：I want to deactivate my account.

文本 B：I wish to inactivate my account.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：deactivate相比原文inactivate更加地道规范
## M1-037 · banking77/train/9270

原标签：activate_my_card

原文：I have a new card and need to activate it

文本 A：I have a new card and need to enable it

文本 B：I possess a new card and need to activate it

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 否 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本B中I possess a new card极度不自然，口语客户提问不使用possess表达持卡
## M1-039 · banking77/train/3480

原标签：balance_not_updated_after_cheque_or_cash_deposit

原文：I deposited a cheque and its been days and I still haven't received the cash!!

文本 A：I submitted a cheque and its been days and I still haven't received the cash!!

文本 B：I deposited a cheque and its been days and I still haven't got the cash!!

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：submitted a cheque与got the cash均自然通顺
## M1-040 · banking77/train/1721

原标签：pin_blocked

原文：After putting in the wrong PIN too many times, I was blocked. Can you assist me in changing it?

文本 A：After putting in the wrong PIN too many times, I was locked. Can you assist me in changing it?

文本 B：After typing in the wrong PIN too many times, I was blocked. Can you assist me in changing it?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 否 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本A中I was locked用法错误，应为I was locked out或my account/PIN was locked
## M1-041 · banking77/train/6755

原标签：reverted_card_payment?

原文：The card payment didn't work

文本 A：The card purchase didn't work

文本 B：The card payment didn't succeed

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：card purchase与didn't succeed均保留支付未成功意图
## M1-043 · banking77/train/6964

原标签：change_pin

原文：I want to create a new pin.

文本 A：I want to create a new passcode.

文本 B：I want to set a new pin.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：passcode与set a new pin均自然准确
## M1-044 · banking77/train/336

原标签：exchange_rate

原文：Please explain your exchange rate policy.

文本 A：Please explain your conversion rate policy.

文本 B：Please describe your exchange rate policy.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：conversion rate与describe policy均为标准表达
## M1-045 · banking77/train/9320

原标签：activate_my_card

原文：The activation process did not work, please help?

文本 A：The activation procedure did not work, please help?

文本 B：The activation process did not work, please assist?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：procedure与assist保持激活失败求助意图
## M1-046 · banking77/train/3371

原标签：balance_not_updated_after_cheque_or_cash_deposit

原文：I updated by balance with a cheque yesterday but it isn't showing on my account. Did something go wrong?

文本 A：I updated by balance with a check yesterday but it isn't showing on my account. Did something go wrong?

文本 B：I updated by balance with a cheque yesterday but it isn't showing on my account. Did something go awry?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：check与go awry均自然，保持支票未到账意图
## M1-050 · banking77/train/270

原标签：card_linking

原文：Can you tell me how to get my card on the app?

文本 A：Can you tell me how to get my bankcard on the app?

文本 B：Can you tell me how to add my card on the app?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：add my card准确对应卡片绑定(card linking)操作
## M1-052 · banking77/train/4157

原标签：visa_or_mastercard

原文：I'm wanting to get a Visa card today.

文本 A：I'm wanting to obtain a Visa card today.

文本 B：I'm hoping to get a Visa card today.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：hoping to get比原文口语I'm wanting更加自然
## M1-054 · banking77/train/3216

原标签：top_up_reverted

原文：What should I do? I topped off but the app reversed it.

文本 A：What should I do? I topped off but the app reverted it.

文本 B：What should I do? I topped off but the application reversed it.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：reverted为充值被撤回的标准金融术语，完全吻合标签
## M1-055 · banking77/train/3330

原标签：balance_not_updated_after_cheque_or_cash_deposit

原文：i need help. I updated my balance with a check yesterday and Im not seeing it. Can you look at my account to see what went wrong?

文本 A：i need help. I updated my balance with a cheque yesterday and Im not seeing it. Can you look at my account to see what went wrong?

文本 B：i need help. I updated my balance with a check yesterday and Im not seeing it. Can you look at my account to see what went awry?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：英美拼写与自然同义词替换
## M1-056 · banking77/train/4713

原标签：direct_debit_payment_not_recognised

原文：I don't know where this debit transaction on my statement came from.

文本 A：I don't understand where this debit transaction on my statement came from.

文本 B：I don't know where this debit payment on my statement came from.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：debit payment准确对应直接借记扣款意图
## M1-057 · banking77/train/1019

原标签：card_delivery_estimate

原文：Can I choose when my card is delivered?

文本 A：Can I choose when my card is received?

文本 B：Can I choose when my bankcard is delivered?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：询问是否能选择交付时间，两臂均保留意图
## M1-058 · banking77/train/3666

原标签：edit_personal_details

原文：How can I change my name because I just got married?

文本 A：How can I update my name because I just got married?

文本 B：How can I change my name because I recently got married?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：update my name与recently got married极其自然地道
## M1-059 · banking77/train/2023

原标签：pending_top_up

原文：I'm pretty sure my top up is still pending.

文本 A：I'm pretty certain my top up is still pending.

文本 B：I'm pretty sure my top up is still unprocessed.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 不确定 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：unprocessed偏指未被系统处理，与金融交易pending（处理中）概念略有偏差
## M1-063 · banking77/train/307

原标签：exchange_rate

原文：I need to know what your exchange rates are.

文本 A：I need to know what your conversion rates are.

文本 B：I need to understand what your exchange rates are.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：conversion rates为外汇兑换率标准表达
## M1-064 · banking77/train/3842

原标签：why_verify_identity

原文：Why do you need me to do an identity check?

文本 A：Why do you require me to do an identity check?

文本 B：Why do you need me to do an identification check?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：require与identification check完全等价且自然
## M1-065 · banking77/train/6433

原标签：transaction_charged_twice

原文：Hi. I noticed I have a duplicate charge which I have not made twice. Who should I talk to take care of this?

文本 A：Hi. I noticed I have a duplicate charge which I have not performed twice. Who should I talk to take care of this?

文本 B：Hi. I noticed I have a repeated charge which I have not made twice. Who should I talk to take care of this?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 不确定 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：performed a charge搭配偏生硬（charge通常为made或incurred），repeated charge自然
## M1-067 · banking77/train/6533

原标签：verify_source_of_funds

原文：I would like to verify my source of funds.

文本 A：I would like to verify my source of money.

文本 B：I would prefer to verify my source of funds.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 否 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本B中prefer引入了偏好选择含义（更倾向于），改变了原文单纯表达意愿的请求意图
## M1-072 · banking77/train/6462

原标签：transaction_charged_twice

原文：I have been double charged for a couple of things this week, and would appreciate a refund for the duplicate charges.

文本 A：I have been double charged for a couple of things this week, and would appreciate a refund for the repeated charges.

文本 B：I have been double charged for a couple of things this week, and would welcome a refund for the duplicate charges.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：repeated与welcome a refund均礼貌地道
## M1-073 · banking77/train/534

原标签：card_payment_wrong_exchange_rate

原文：When I purchased a product in another country the currency exchange was incorrect.

文本 A：When I purchased a item in another country the currency exchange was incorrect.

文本 B：When I purchased a product in another country the currency exchange was wrong.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 否 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本A出现语法错误：a item应为an item，违背英语基本冠词规则
## M1-074 · banking77/train/5968

原标签：declined_card_payment

原文：I just got my new card and tried using it and it was declined?

文本 A：I just got my new card and tried using it and it was rejected?

文本 B：I just got my new card and attempted using it and it was declined?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：rejected与attempted using均自然保留新卡被拒意图
## M1-075 · banking77/train/7894

原标签：getting_spare_card

原文：If I get another card, what kind of fees am I looking at?

文本 A：If I get another card, what kind of charges am I looking at?

文本 B：If I get spare card, what kind of fees am I looking at?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 否 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本B出现语法错误：get spare card缺失单数可数名词冠词a（应为get a spare card）
## M1-076 · banking77/train/3820

原标签：why_verify_identity

原文：Do you think you're asking for too much information about me?

文本 A：Do you believe you're asking for too much information about me?

文本 B：Do you think you're asking for too much details about me?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 否 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本B语法错误：details为复数可数名词，不可与much搭配（应为too many details）
## M1-077 · banking77/train/7900

原标签：getting_spare_card

原文：How do I order another card?

文本 A：How do I order another bankcard?

文本 B：How do I request another card?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：bankcard与request another card自然通顺，申办备用卡意图明确
## M1-078 · banking77/train/5759

原标签：Refund_not_showing_up

原文：I requested a refund from Next because the dress I purchased had three arm holes. They said they sent the refund last Wednesday which is nearly a week ago and I still don't have it. Why not?

文本 A：I requested a reimbursement from Next because the dress I purchased had three arm holes. They said they sent the refund last Wednesday which is nearly a week ago and I still don't have it. Why not?

文本 B：I requested a refund from Next because the dress I bought had three arm holes. They said they sent the refund last Wednesday which is nearly a week ago and I still don't have it. Why not?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：reimbursement与bought均自然保留商户退款未收到意图
## M1-080 · banking77/train/8201

原标签：virtual_card_not_working

原文：I tried to use my disposable virtual card to pay a subscription to the gym and it got rejected. Any ideas why?

文本 A：I tried to use my disposable virtual card to pay a subscription to the gym and it got declined. Any ideas why?

文本 B：I tried to use my disposable virtual card to pay a membership to the gym and it got rejected. Any ideas why?

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：declined与gym membership自然贴切，虚拟卡订阅被拒意图一致