# Debug Samples For RecSys

Debug-датасет позволяет смотреть динамику на заранее выбранных сценариях:


- `cold_user` — пользователь с очень короткой историей взаимодействий (нижний квантиль по `user_prior_interactions`).
- `tail_item` — редкий товар в конкретном месяце (нижний квантиль по `item_popularity_in_month`).
- `head_item` — популярный товар в конкретном месяце (верхний квантиль по `item_popularity_in_month`).
- `short_history_user` — пользователь с малым числом уникальных товаров в истории (нижний квантиль по `user_prior_unique_items`).
- `ambiguous_user` — пользователь с высокой диверсификацией истории (верхний квантиль по `user_prior_diversity`).
- `easy_control` — «контрольный простой» кейс: популярный товар (`head`) и пользователь не cold.

## Функция `build_debug_dataset(...)`:

1. Берет позитивные взаимодействия (`count_col > 0`).
2. Считает признаки сложности (история пользователя, популярность айтема в месяце, diversity).
3. Стратифицирует выборку по кейсам сложности (управляемо через `per_case`).
4. Для каждого сэмпла собирает кандидатов для ранжирования:
   - `target_item_id` (первый в списке)
   - hard negatives (популярные, same-category, случайные).

На выходе — датафрейм, где одна строка = один debug-кейс ранжирования.

## Основные поля результата

- `debug_id` — стабильный ID кейса
- `stratum` — основной тип кейса
- `target_item_id`
- `candidates` — список кандидатов (первый элемент — target)
- `candidate_count`
- диагностические признаки: `user_prior_*`, `item_popularity_in_month`
- бинарные флаги для группировки:
  - `is_cold_user`
  - `is_tail_item`
  - `is_head_item`
  - `is_short_history_user`
  - `is_ambiguous_user`
  - `is_easy_control`

Флаги рассчитываются на основе квантилей, посчитанных по всем позитивным взаимодействиям (`count_col > 0`):

- `cold_user` threshold: `q(user_prior_interactions, cold_user_quantile)` (по умолчанию `q=0.2`)
- `tail_item` threshold: `q(item_popularity_in_month, tail_item_quantile)` (по умолчанию `q=0.2`)
- `head_item` threshold: `q(item_popularity_in_month, head_item_quantile)` (по умолчанию `q=0.8`)
- `short_history_user` threshold: `q(user_prior_unique_items, short_history_quantile)` (по умолчанию `q=0.2`)
- `ambiguous_user` threshold: `q(user_prior_diversity, ambiguous_quantile)` (по умолчанию `q=0.8`)

Где:
- `user_prior_interactions` — число прошлых взаимодействий пользователя до текущего кейса
- `user_prior_unique_items` — число уникальных товаров в прошлой истории пользователя
- `user_prior_diversity = user_prior_unique_items / max(user_prior_interactions, 1)`
- `item_popularity_in_month` — суммарный `count_col` для `(month, item)`

Логика флагов:

- `is_cold_user`: `user_prior_interactions <= cold_user_threshold`
- `is_tail_item`: `item_popularity_in_month <= tail_item_threshold`
- `is_head_item`: `item_popularity_in_month >= head_item_threshold`
- `is_short_history_user`: `user_prior_unique_items <= short_history_threshold`
- `is_ambiguous_user`: `user_prior_diversity >= ambiguous_threshold`
- `is_easy_control`: `is_head_item == True` и пользователь не cold (`user_prior_interactions > cold_user_threshold`)
- `stratum` — это основной класс кейса (каждая строка попадает только в один `stratum` в рамках выборки).
- Флаги — мульти-лейбл разметка: одна строка может иметь несколько `is_* = True`.