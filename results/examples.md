# Validation recommendation examples

Validation split: training weeks 11-20; truth weeks 21-30. `hit` means the item is in the user's validation truth; `repeat` means the user watched it in training weeks 11-20.

## Heavy viewer

- user_id: `13571199046623737884`
- why chosen: top decile (threshold 254); selected 634 training programmes; truth size 246
- training programmes: 634
- validation truth size: 246

### Training: 5 most-watched programmes by distinct days

| rank | tv_show_id | title | category | distinct_days |
|---:|---:|---|---|---:|
| 1 | 240081 | Факты. День. | Инфо | 26 |
| 2 | 2400480 | ТСН. | Инфо | 26 |
| 3 | 2400508 | Сегодня. | Инфо | 22 |
| 4 | 6500479 | 'Мир наизнанку'. | Познавательное | 21 |
| 5 | 400335 | 'Хата на тата', 6 сезон, 3 эп. | Развлечения | 16 |

### popularity

Hits@5: 4/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 2400480 | ТСН. | Инфо | yes | yes |
| 2 | 240081 | Факты. День. | Инфо | yes | yes |
| 3 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | yes |
| 4 | 10002514 | Т/с 'Кухня'. | Сериалы | no | yes |
| 5 | 2400508 | Сегодня. | Инфо | yes | yes |

### content_tfidf

Hits@5: 1/5; repeats in Top-5: 3/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 200477 | 'Завтрак. Выходной'. | Развлечения | no | yes |
| 2 | 400357 | 'Холостяк', 10 сезон, 7 эп., 2 ч. | Развлечения | yes | yes |
| 3 | 400373 | 'Холостяк'. Реалити-шоу, 7 сезон, 4 эп. | Развлечения | no | no |
| 4 | 500315 | Ток-шоу 'Касается каждого'. | Развлечения | no | yes |
| 5 | 6000473 | Половинки, 3 сезон, 3 эп. | Развлечения | no | no |

### mf_k16

Hits@5: 3/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | yes |
| 2 | 12001682 | М/с 'Губка Боб Квадратные Штаны', 249 с. 'Племянницы Сэнди / Бравые охранники'. | Детям | no | yes |
| 3 | 2500430 | 'Затерянный мир'. | Познавательное | yes | yes |
| 4 | 240081 | Факты. День. | Инфо | yes | yes |
| 5 | 5800509 | Реальная мистика. | Развлечения | no | yes |

### hist_pop_w0.30

Hits@5: 5/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 2400480 | ТСН. | Инфо | yes | yes |
| 2 | 240081 | Факты. День. | Инфо | yes | yes |
| 3 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | yes |
| 4 | 2400508 | Сегодня. | Инфо | yes | yes |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | yes | yes |

### final mf_blend (a=0.30, b=0.02, pop=0.68)

Hits@5: 4/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | yes |
| 2 | 240081 | Факты. День. | Инфо | yes | yes |
| 3 | 2400480 | ТСН. | Инфо | yes | yes |
| 4 | 12001682 | М/с 'Губка Боб Квадратные Штаны', 249 с. 'Племянницы Сэнди / Бравые охранники'. | Детям | no | yes |
| 5 | 2500430 | 'Затерянный мир'. | Познавательное | yes | yes |

## Light viewer

- user_id: `12453026253397466416`
- why chosen: bottom decile (threshold 56); selected 5 training programmes (>=3); truth size 95
- training programmes: 5
- validation truth size: 95

### Training: 5 most-watched programmes by distinct days

| rank | tv_show_id | title | category | distinct_days |
|---:|---:|---|---|---:|
| 1 | 2400508 | Сегодня. | Инфо | 2 |
| 2 | 200337 | 'Утро с 'Интером'. | Развлечения | 1 |
| 3 | 2400480 | ТСН. | Инфо | 1 |
| 4 | 5200303 | 'Следствие вели...' с Леонидом Каневским. | Познавательное | 1 |
| 5 | 6700482 | 'Секретные материалы'. | Другое | 1 |

### popularity

Hits@5: 5/5; repeats in Top-5: 2/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 2400480 | ТСН. | Инфо | yes | yes |
| 2 | 240081 | Факты. День. | Инфо | yes | no |
| 3 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no |
| 4 | 10002514 | Т/с 'Кухня'. | Сериалы | yes | no |
| 5 | 2400508 | Сегодня. | Инфо | yes | yes |

### content_tfidf

Hits@5: 2/5; repeats in Top-5: 1/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 2400508 | Сегодня. | Инфо | yes | yes |
| 2 | 5900256 | Утро с Украиной. | Инфо | no | no |
| 3 | 5900276 | 'Центральное телевидение' с Вадимом Такменёвым. | Инфо | no | no |
| 4 | 5900275 | 'Утро России'. | Инфо | no | no |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | yes | no |

### mf_k16

Hits@5: 5/5; repeats in Top-5: 4/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 2400480 | ТСН. | Инфо | yes | yes |
| 2 | 2400508 | Сегодня. | Инфо | yes | yes |
| 3 | 5200303 | 'Следствие вели...' с Леонидом Каневским. | Познавательное | yes | yes |
| 4 | 1100411 | 'Вещдок. Особый случай. Чужое богатство'. | Познавательное | yes | no |
| 5 | 200337 | 'Утро с 'Интером'. | Развлечения | yes | yes |

### hist_pop_w0.30

Hits@5: 5/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 2400508 | Сегодня. | Инфо | yes | yes |
| 2 | 2400480 | ТСН. | Инфо | yes | yes |
| 3 | 6700482 | 'Секретные материалы'. | Другое | yes | yes |
| 4 | 5200303 | 'Следствие вели...' с Леонидом Каневским. | Познавательное | yes | yes |
| 5 | 200337 | 'Утро с 'Интером'. | Развлечения | yes | yes |

### final mf_blend (a=0.30, b=0.02, pop=0.68)

Hits@5: 5/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | hit | repeat |
|---:|---:|---|---|:---:|:---:|
| 1 | 2400508 | Сегодня. | Инфо | yes | yes |
| 2 | 2400480 | ТСН. | Инфо | yes | yes |
| 3 | 5200303 | 'Следствие вели...' с Леонидом Каневским. | Познавательное | yes | yes |
| 4 | 200337 | 'Утро с 'Интером'. | Развлечения | yes | yes |
| 5 | 6700482 | 'Секретные материалы'. | Другое | yes | yes |

## Repeat and best reranker (strict truth)

Repeat model: `t=0.30`, fit weeks 11-20, strict-popularity fill. Best reranker: LightGBM, validation strict MAP@5 0.58578. Flags: `✓` = in the user's strict weeks 21-30 truth; `R` = watched in weeks 11-20; `—` = neither. Strict truth programmes differ from the old-truth `validation truth size` values in the sections above.

| viewer | user_id | strict truth programmes |
|---|---|---:|
| Heavy | `13571199046623737884` | 11 |
| Light | `12453026253397466416` | 14 |

### Heavy viewer

- user_id: `13571199046623737884`
- strict truth programmes: 11

#### repeat (t=0.30, fit weeks 11-20)

Strict hits@5: 1/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | flags |
|---:|---:|---|---|:---:|
| 1 | 12001682 | М/с 'Губка Боб Квадратные Штаны', 249 с. 'Племянницы Сэнди / Бравые охранники'. | Детям | R |
| 2 | 240081 | Факты. День. | Инфо | ✓ R |
| 3 | 2400480 | ТСН. | Инфо | R |
| 4 | 2400508 | Сегодня. | Инфо | R |
| 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | R |

#### best reranker: LightGBM

Strict hits@5: 1/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | flags |
|---:|---:|---|---|:---:|
| 1 | 240081 | Факты. День. | Инфо | ✓ R |
| 2 | 200337 | 'Утро с 'Интером'. | Развлечения | R |
| 3 | 2400480 | ТСН. | Инфо | R |
| 4 | 2400467 | 'Викна-Новости'. | Инфо | R |
| 5 | 10003014 | Т/с 'Отдел 44', 36 с. | Сериалы | R |

### Light viewer

- user_id: `12453026253397466416`
- strict truth programmes: 14

#### repeat (t=0.30, fit weeks 11-20)

Strict hits@5: 4/5; repeats in Top-5: 5/5

| rank | tv_show_id | title | category | flags |
|---:|---:|---|---|:---:|
| 1 | 2400480 | ТСН. | Инфо | ✓ R |
| 2 | 2400508 | Сегодня. | Инфо | ✓ R |
| 3 | 6700482 | 'Секретные материалы'. | Другое | R |
| 4 | 200337 | 'Утро с 'Интером'. | Развлечения | ✓ R |
| 5 | 5200303 | 'Следствие вели...' с Леонидом Каневским. | Познавательное | ✓ R |

#### best reranker: LightGBM

Strict hits@5: 3/5; repeats in Top-5: 2/5

| rank | tv_show_id | title | category | flags |
|---:|---:|---|---|:---:|
| 1 | 2400480 | ТСН. | Инфо | ✓ R |
| 2 | 2400508 | Сегодня. | Инфо | ✓ R |
| 3 | 2400467 | 'Викна-Новости'. | Инфо | — |
| 4 | 10002514 | Т/с 'Кухня'. | Сериалы | — |
| 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | ✓ |

