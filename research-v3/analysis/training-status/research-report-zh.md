# 文本分类研究证据报告

状态：未完成；缺失门和证据保留为待完成。
Scope: partial_preparation_or_training

原因状态：unresolved。软件完成不等于真实研究完成。

不跨任务平均F1/AP；各数据集分别报告。

| Gate | Status |
| --- | --- |
| G0 | verified |
| G1 | verified |
| G2 | pending / invalid |
| G3 | pending / invalid |
| G4 | pending / invalid |
| G5 | pending / invalid |

Anchors: 36/39; candidates: 216/234.
Physical checkpoints: 0/2859.

## 数据与支持数

### ag_news (single_label)
```json
{
  "task": "single_label",
  "label_count": 4,
  "partitions": {
    "train": {
      "rows": 118784,
      "label_support": [
        29721,
        29722,
        29685,
        29656
      ],
      "ap_mask": [
        0,
        1,
        2,
        3
      ]
    },
    "select": {
      "rows": 600,
      "label_support": [
        154,
        145,
        153,
        148
      ],
      "ap_mask": [
        0,
        1,
        2,
        3
      ]
    },
    "discovery": {
      "rows": 400,
      "label_support": [
        96,
        105,
        97,
        102
      ],
      "ap_mask": [
        0,
        1,
        2,
        3
      ]
    },
    "test": {
      "rows": 7600,
      "label_support": [
        1900,
        1900,
        1900,
        1900
      ],
      "ap_mask": [
        0,
        1,
        2,
        3
      ]
    },
    "quarantine": {
      "rows": 216,
      "label_support": [
        29,
        28,
        65,
        94
      ],
      "ap_mask": [
        0,
        1,
        2,
        3
      ]
    }
  }
}
```

### banking77 (single_label)
```json
{
  "task": "single_label",
  "label_count": 77,
  "partitions": {
    "train": {
      "rows": 8058,
      "label_support": [
        129,
        88,
        101,
        69,
        103,
        137,
        145,
        126,
        127,
        105,
        48,
        123,
        90,
        113,
        89,
        151,
        136,
        135,
        49,
        143,
        128,
        96,
        69,
        29,
        104,
        123,
        139,
        106,
        146,
        97,
        97,
        97,
        90,
        96,
        133,
        110,
        102,
        79,
        86,
        105,
        80,
        66,
        97,
        96,
        85,
        129,
        115,
        121,
        120,
        91,
        77,
        130,
        137,
        129,
        105,
        88,
        89,
        92,
        92,
        117,
        79,
        118,
        83,
        141,
        137,
        91,
        137,
        104,
        82,
        84,
        91,
        102,
        33,
        109,
        97,
        144,
        131
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76
      ]
    },
    "select": {
      "rows": 964,
      "label_support": [
        15,
        11,
        12,
        8,
        12,
        17,
        18,
        15,
        15,
        12,
        4,
        15,
        11,
        13,
        11,
        18,
        16,
        16,
        6,
        17,
        16,
        12,
        8,
        3,
        12,
        15,
        17,
        13,
        18,
        12,
        12,
        12,
        11,
        11,
        16,
        13,
        12,
        9,
        10,
        12,
        9,
        8,
        12,
        12,
        10,
        15,
        14,
        14,
        14,
        11,
        9,
        16,
        16,
        16,
        12,
        10,
        11,
        11,
        11,
        14,
        9,
        14,
        10,
        17,
        17,
        11,
        17,
        12,
        10,
        10,
        11,
        12,
        4,
        13,
        12,
        18,
        16
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76
      ]
    },
    "discovery": {
      "rows": 964,
      "label_support": [
        15,
        11,
        12,
        8,
        12,
        17,
        18,
        15,
        15,
        12,
        5,
        15,
        11,
        13,
        11,
        18,
        16,
        16,
        6,
        17,
        16,
        12,
        8,
        3,
        11,
        15,
        17,
        13,
        18,
        12,
        12,
        12,
        11,
        11,
        16,
        13,
        12,
        9,
        10,
        12,
        9,
        8,
        12,
        12,
        10,
        15,
        14,
        14,
        14,
        11,
        9,
        16,
        16,
        16,
        12,
        10,
        11,
        11,
        11,
        14,
        9,
        14,
        10,
        17,
        17,
        11,
        17,
        12,
        10,
        10,
        11,
        12,
        4,
        13,
        12,
        18,
        16
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76
      ]
    },
    "test": {
      "rows": 3079,
      "label_support": [
        40,
        40,
        40,
        39,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76
      ]
    },
    "quarantine": {
      "rows": 18,
      "label_support": [
        0,
        0,
        1,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        1,
        0,
        2,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "ap_mask": [
        2,
        3,
        10,
        14,
        21,
        22,
        24,
        27,
        34,
        35,
        49,
        64
      ]
    }
  }
}
```

### go_emotions (multilabel)
```json
{
  "task": "multilabel",
  "label_count": 28,
  "partitions": {
    "train": {
      "rows": 43064,
      "label_support": [
        4104,
        2317,
        1558,
        2458,
        2926,
        1077,
        1361,
        2182,
        641,
        1264,
        2014,
        787,
        302,
        835,
        591,
        2589,
        77,
        1430,
        2048,
        163,
        1572,
        110,
        1106,
        153,
        543,
        1316,
        1054,
        14142
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27
      ]
    },
    "select": {
      "rows": 3237,
      "label_support": [
        279,
        184,
        114,
        183,
        226,
        96,
        89,
        170,
        44,
        91,
        190,
        59,
        25,
        55,
        45,
        200,
        8,
        97,
        155,
        11,
        125,
        7,
        76,
        11,
        40,
        87,
        78,
        1064
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27
      ]
    },
    "discovery": {
      "rows": 2158,
      "label_support": [
        208,
        119,
        79,
        120,
        169,
        55,
        61,
        76,
        32,
        71,
        102,
        38,
        10,
        38,
        45,
        149,
        5,
        74,
        94,
        10,
        84,
        8,
        51,
        7,
        28,
        55,
        51,
        698
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27
      ]
    },
    "test": {
      "rows": 5401,
      "label_support": [
        501,
        264,
        198,
        320,
        350,
        134,
        153,
        283,
        83,
        151,
        267,
        122,
        37,
        99,
        78,
        349,
        5,
        160,
        237,
        23,
        185,
        16,
        145,
        11,
        56,
        154,
        141,
        1778
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27
      ]
    },
    "quarantine": {
      "rows": 403,
      "label_support": [
        30,
        11,
        11,
        12,
        16,
        13,
        9,
        12,
        1,
        6,
        8,
        7,
        1,
        25,
        5,
        85,
        1,
        24,
        42,
        1,
        10,
        1,
        4,
        0,
        2,
        13,
        6,
        90
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        24,
        25,
        26,
        27
      ]
    }
  }
}
```

### imdb (single_label)
```json
{
  "task": "single_label",
  "label_count": 2,
  "partitions": {
    "train": {
      "rows": 23779,
      "label_support": [
        11836,
        11943
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "select": {
      "rows": 600,
      "label_support": [
        309,
        291
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "discovery": {
      "rows": 400,
      "label_support": [
        191,
        209
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "test": {
      "rows": 24799,
      "label_support": [
        12360,
        12439
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "quarantine": {
      "rows": 422,
      "label_support": [
        304,
        118
      ],
      "ap_mask": [
        0,
        1
      ]
    }
  }
}
```

### rotten_tomatoes (single_label)
```json
{
  "task": "single_label",
  "label_count": 2,
  "partitions": {
    "train": {
      "rows": 8530,
      "label_support": [
        4265,
        4265
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "select": {
      "rows": 600,
      "label_support": [
        310,
        290
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "discovery": {
      "rows": 400,
      "label_support": [
        190,
        210
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "test": {
      "rows": 1066,
      "label_support": [
        533,
        533
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "quarantine": {
      "rows": 66,
      "label_support": [
        33,
        33
      ],
      "ap_mask": [
        0,
        1
      ]
    }
  }
}
```

### yahoo_answers_topics (single_label)
```json
{
  "task": "single_label",
  "label_count": 10,
  "partitions": {
    "train": {
      "rows": 1383616,
      "label_support": [
        138438,
        138373,
        138232,
        138358,
        138362,
        138399,
        138152,
        138436,
        138463,
        138403
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
      ]
    },
    "select": {
      "rows": 9989,
      "label_support": [
        1000,
        998,
        995,
        1000,
        999,
        1000,
        998,
        1000,
        999,
        1000
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
      ]
    },
    "discovery": {
      "rows": 4993,
      "label_support": [
        500,
        500,
        498,
        499,
        499,
        500,
        498,
        500,
        500,
        499
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
      ]
    },
    "test": {
      "rows": 59998,
      "label_support": [
        6000,
        6000,
        6000,
        6000,
        6000,
        6000,
        5998,
        6000,
        6000,
        6000
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
      ]
    },
    "quarantine": {
      "rows": 1404,
      "label_support": [
        62,
        129,
        275,
        143,
        140,
        101,
        354,
        64,
        38,
        98
      ],
      "ap_mask": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
      ]
    }
  }
}
```

### yelp (single_label)
```json
{
  "task": "single_label",
  "label_count": 2,
  "partitions": {
    "train": {
      "rows": 558896,
      "label_support": [
        279429,
        279467
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "select": {
      "rows": 600,
      "label_support": [
        294,
        306
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "discovery": {
      "rows": 400,
      "label_support": [
        206,
        194
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "test": {
      "rows": 38000,
      "label_support": [
        19000,
        19000
      ],
      "ap_mask": [
        0,
        1
      ]
    },
    "quarantine": {
      "rows": 104,
      "label_support": [
        71,
        33
      ],
      "ap_mask": [
        0,
        1
      ]
    }
  }
}
```

## 经重建的模型结果
| Dataset | Model | Condition | Endpoint | Value | Exposure |
| --- | --- | --- | --- | --- | --- |
| pending | — | — | — | — | — |

各run的完整类别/标签支持数、F1/AP与混淆矩阵保存在completion.json的results中。

## 统计与失败/机制证据
pending；没有经过预测重建的统计结论。

### 失败迁移与人工干预
```json
{
  "failure_trajectories": {},
  "audit": null,
  "confirmation": [],
  "claims": []
}
```

## 证据缺口
- G2 / anchor:select:yahoo_answers_topics:bilstm: `selections.json` — Anchor search remains pending: select:yahoo_answers_topics:bilstm
- G2 / anchor:select:yahoo_answers_topics:distilbert: `selections.json` — Anchor search remains pending: select:yahoo_answers_topics:distilbert
- G2 / anchor:select:yahoo_answers_topics:modernbert: `selections.json` — Anchor search remains pending: select:yahoo_answers_topics:modernbert
- G2 / pilot:Bank_M1_SVM: `pilots/Bank_M1_SVM/results.json` — Real human D pilot is pending: Bank_M1_SVM
- G2 / pilot:Yelp_M4_DistilBERT: `pilots/Yelp_M4_DistilBERT/results.json` — Real human D pilot is pending: Yelp_M4_DistilBERT
- G2 / pilot:Yahoo_M4_DistilBERT: `pilots/Yahoo_M4_DistilBERT/results.json` — Real human D pilot is pending: Yahoo_M4_DistilBERT
- G2 / global_selection: `selection.lock.json` — A verified global selection freeze is required
- G3 / global_evaluation: `evaluation.freeze.json` — A verified global evaluation freeze is required
- G3 / run_manifest: `run_manifest.json` — Required evidence missing: run_manifest.json
- G3 / all_planned_training_coverage: `run_manifest.json` — Resolved manifest still has incomplete selections/pilots
- G4 / evidence_routing_index: `analysis/research-evidence.json` — Required evidence missing: analysis/research-evidence.json
- G4 / real_A_double_review: `audit/reviews.json` — Authenticated evaluation freeze and byte-bound evidence index are prerequisites
- G4 / paired_confirmation: `confirmation/` — Authenticated evaluation freeze and byte-bound evidence index are prerequisites
- G4 / all_statistical_families: `analysis/statistics/` — Authenticated evaluation freeze and byte-bound evidence index are prerequisites
- G5 / report_evidence: `analysis/research-evidence.json` — Authenticated evaluation freeze and byte-bound evidence index are prerequisites

## 资源成本
```json
[]
```

## 限制与相关工作
- Conditional test-sampling intervals describe saved fitted models; few seeds limit generalization.
- Task type is confounded with corpus/style/annotation; external pretraining exposure is not fully auditable.
- Historical Q remains previously seen exploratory evidence; automated records do not replace genuine human review.
- Unsupported explanations remain unresolved; auxiliary mechanism results do not replace primary families.
相关工作比较须由已绑定的矩阵给出；缺失时保持pending。
