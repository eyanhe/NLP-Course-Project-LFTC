# M1 独立人审表 B

共 56 对 AI 编辑草稿，来自原预注册候选顺序的前80条。实际完成真人复核：0。目标为50个双人同意的有效对；当前不保证达到。

复核者真实姓名/代号：复核员B；日期：2026-11-08；复核时是否见过这些样本的模型预测：未见过。

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
| A 自然通顺，未新增歧义 | 不确定 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：文本A中extra surcharge存在轻度语义冗余；文本B通顺，原标签符合槽位
## M1-002 · banking77/train/3484

原标签：card_payment_not_recognised

原文：There's a payment with my card that I definitely didn't do myself, never seen that name before.

文本 A：There's a payment with my bankcard that I definitely didn't do myself, never seen that name before.

文本 B：There's a transaction with my card that I definitely didn't do myself, never seen that name before.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 是 |
| B 保持原文含义、标签和关键槽位 | 否 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：B将payment替换为transaction语义范围过宽，削弱了card payment意图
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
| B 自然通顺，未新增歧义 | 不确定 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：B替换wish偏离当下咨询语气，个人账户A自然且保持原意
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

备注/拒绝原因/修改建议：含义与意图均保持，nations在银行业务中稍显宏大但通顺无歧义
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

备注/拒绝原因/修改建议：缩写展开自然，量词替换合适
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

备注/拒绝原因/修改建议：软件功能同义替换，自然通顺
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
| B 自然通顺，未新增歧义 | 不确定 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：arbitrarily在口语客户提问中略显书面化，但语义完全一致
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

备注/拒绝原因/修改建议：语义保持良好，替换为内容形容词
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

备注/拒绝原因/修改建议：recipient准确度高，双方均自然
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

备注/拒绝原因/修改建议：动词替换自然，槽位完整
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

备注/拒绝原因/修改建议：made transaction与did transfer搭配均更自然
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
| B 自然通顺，未新增歧义 | 不确定 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：throwaway偏口语俚俗，不可替代disposable virtual card金融术语；possess过于正式
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

备注/拒绝原因/修改建议：A语法错误（uncountable noun money配合动词原形come），需拒绝
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

备注/拒绝原因/修改建议：含义保持完整，自然无歧义
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

备注/拒绝原因/修改建议：外币业务支持咨询，替换通顺
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

备注/拒绝原因/修改建议：虽原句有trasfer拼写错误，两臂忠实保留且替换自然
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

备注/拒绝原因/修改建议：语义保持，对账单未显示退款意图明确
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

备注/拒绝原因/修改建议：意图与关键槽位完整保留
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

备注/拒绝原因/修改建议：英美拼写与邮寄动词替换，自然度极高
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

备注/拒绝原因/修改建议：意图保持清晰，内容词替换合规
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

备注/拒绝原因/修改建议：币种转换意图明确，替换词准确
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

备注/拒绝原因/修改建议：动作与诉求均自然通顺
## M1-027 · banking77/train/3682

原标签：edit_personal_details

原文：I just had a wedding and I want to take my spouses name.

文本 A：I just had a wedding and I wish to take my spouses name.

文本 B：I just had a wedding and I want to take my spouses surname.

| 复核项目 | 判断 |
| --- | --- |
| 原标签与原文相符，且意图可明确判断 | 是 |
| A 保持原文含义、标签和关键槽位 | 不确定 |
| B 保持原文含义、标签和关键槽位 | 是 |
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：wish语气偏委婉愿望，surname更为精确
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
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：cap和include均能表达是否存在交易限额的意图
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

备注/拒绝原因/修改建议：卡支付取消/撤销原因咨询，语义保持完好
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

备注/拒绝原因/修改建议：ATM/cashpoint等价，两臂均自然
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
| A 自然通顺，未新增歧义 | 不确定 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：Assist me开头稍显机械但语法正确，rejected非常自然
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

备注/拒绝原因/修改建议：deactivate更符合银行业务系统标准用语
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

备注/拒绝原因/修改建议：possess用词怪异违背日常英语习惯，应拒绝
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

备注/拒绝原因/修改建议：支票入账延迟抱怨，语义与槽位完整
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

备注/拒绝原因/修改建议：I was locked缺失out导致表达残缺不通顺，需拒绝
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

备注/拒绝原因/修改建议：简单句同义替换，两臂均自然
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

备注/拒绝原因/修改建议：密码修改/创建意图保持良好
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

备注/拒绝原因/修改建议：汇率政策咨询，语义对齐
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

备注/拒绝原因/修改建议：同义词替换，自然通顺
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
| B 自然通顺，未新增歧义 | 不确定 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：go awry偏文学色彩，但在此疑问句中依然成立
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

备注/拒绝原因/修改建议：bankcard和add均为恰当内容词，意图明确
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

备注/拒绝原因/修改建议：办卡意图完整，内容词替换合适
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

备注/拒绝原因/修改建议：reverted和application均自然通顺
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

备注/拒绝原因/修改建议：支票充值未到账意图明确，语义保持
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

备注/拒绝原因/修改建议：understand与payment均通顺自然
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
| A 自然通顺，未新增歧义 | 不确定 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：A中choose when my card is received略微偏离交付主动选择，但整体可理解
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

备注/拒绝原因/修改建议：婚后改名意图明确，替换词准确
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
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：certain与unprocessed均能表达充值未结算状态
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

备注/拒绝原因/修改建议：意图保持清晰，替换均为内容词
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

备注/拒绝原因/修改建议：身份核验质询意图保持良好
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
| A 自然通顺，未新增歧义 | 是 |
| B 自然通顺，未新增歧义 | 是 |
| A 替换的是内容词 | 是 |
| B 替换的是内容词 | 是 |

备注/拒绝原因/修改建议：重复扣款投诉意图保留，双方均可理解
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

备注/拒绝原因/修改建议：prefer改变了情态和意图（变为偏好比较），不保持原意
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

备注/拒绝原因/修改建议：重复收费退款诉求，保持完整
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

备注/拒绝原因/修改建议：A冠词错误（a item应为an item），通顺度不达标
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

备注/拒绝原因/修改建议：刷卡被拒事实清楚，同义词准确
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

备注/拒绝原因/修改建议：B残缺冠词（get spare card语法不通顺），予以拒绝
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

备注/拒绝原因/修改建议：B语法错误（much修饰可数复数details），不合规范
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

备注/拒绝原因/修改建议：办卡请求同义替换，槽位完整
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

备注/拒绝原因/修改建议：退款延迟问题，语义与情节完全一致
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

备注/拒绝原因/修改建议：健身房会员/订阅扣款被拒，术语与语境吻合