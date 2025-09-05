# Schema长度与精度校验功能 - 开发实施方案

## 1. 概述

### 1.1. 背景

当前 `validatelite` 系统中的 `SCHEMA` 规则能够校验数据库表的列是否存在，以及列的数据类型是否与预期的通用类型（如 `STRING`, `INTEGER`）匹配。然而，它在设计上有意忽略了具体的物理存储属性，如字符串/二进制的长度、数字的精度（precision）和标度（scale）等。

### 1.2. 目标

本次开发旨在增强现有的 `SCHEMA` 规则，使其能够选择性地校验列的长度和精度信息。此功能必须具备以下特点：

- **方言感知 (Dialect-Aware)**: 所有与特定数据库相关的元数据获取逻辑，必须封装在 `DatabaseDialect` 层，以保证系统的可扩展性。
- **向后兼容 (Backward-Compatible)**: 如果用户提供的规则文件中不包含新的长度/精度属性，系统的行为必须与当前版本完全一致。

## 2. 需求规格

### 2.1. 用户侧规则定义

在用户提供的 `rules.json` 文件中，我们将为 `rules` 数组中的每个列定义对象增加以下可选属性：

| 属性名 | 类型 | 适用用户类型 | 描述 |
| :--- | :--- | :--- | :--- |
| `length` | `integer` | `string`, `binary` | 字符串或二进制类型的最大长度 |
| `precision` | `integer` | `integer`, `float` | 数字类型的总精度（总位数） |
| `scale` | `integer` | `float` | 浮点/定点数的小数位数 |
| `datetime_precision` | `integer` | `datetime` | 日期时间类型的小数秒精度 |

### 2.2. 支持的数据类型

本次功能增强将覆盖以下用户定义类型：

- **需要改造**: `string`, `integer`, `float`, `datetime`, 以及新增的 `binary`。
- **不涉及**: `boolean`, `date`。

### 2.3. 向后兼容性

此为强制性要求。当 `ValidityExecutor` 执行 `SCHEMA` 规则时，如果规则参数中不包含上述新属性，则其校验逻辑和结果必须与现有系统完全相同，仅校验列存在性和通用类型。

## 3. 设计方案

### 3.1. 架构核心思想

严格遵循现有分层架构，将数据库差异处理的复杂性限制在 `DatabaseDialect` 层。`ValidityExecutor` 作为核心执行器，保持通用性，它仅处理由 `QueryExecutor` 和 `DatabaseDialect` 提供的、经过标准化的元数据，而不直接解析特定数据库的类型字符串。

### 3.2. 分层实施细节

#### 3.2.1. `QueryExecutor` 层 - 信息传递

- **涉及模块**: `shared/database/query_executor.py`
- **涉及模型**: `QueryExecutor.get_column_list` 方法。
- **逻辑描述**: 修改此方法内部的“标准化结果格式”逻辑。在遍历从数据库查询到的原始列信息时，除了现有的 `name`, `type` 等字段，还需从原始结果 `col` 中提取 `character_maximum_length`, `numeric_precision`, `numeric_scale` 等字段，并将它们作为顶级键添加到返回的标准化字典 `std_col` 中。如果原始结果中不存在这些键（例如SQLite），则对应的值为 `None`。

#### 3.2.2. `ValidityExecutor` 层 - 核心校验

- **涉及模块**: `core/executors/validity_executor.py`
- **涉及模型**: `ValidityExecutor._execute_schema_rule` 方法。
- **逻辑描述**:
  1. 此方法将调用 `QueryExecutor.get_column_list`，获取包含详细元数据（长度、精度等）的列信息字典列表。
  2. 在遍历规则中定义的各列 (`columns_cfg`) 时，执行以下校验算法：
     a. **通用类型校验**: 首先执行现有的 `map_to_datatype` 逻辑，比对通用类型。若失败，则该列校验不通过，终止后续检查。
     b. **长度/精度校验**: 若通用类型校验通过，则继续检查规则参数 `cfg` 中是否包含新属性（如 `length`）。
     c. 如果包含，则将规则中定义的值与从元数据字典中获取的对应值（如 `actual_meta['character_maximum_length']`）进行直接整数比对。若不匹配，则该列校验不通过。
     d. **方言特例处理**: 针对 `SQLite`，由于其元数据查询的特殊性，需要在此方法中增加一个专门的逻辑分支。该分支会检查当前 `dialect` 是否为 `SQLiteDialect`，如果是，则调用一个小的内部辅助函数来从 `type` 字符串（如 `'VARCHAR(50)'`）中解析出长度/精度信息，然后再进行比对。这将所有特殊处理隔离，保持了代码的整洁。
     e. **DateTime精度处理**: 同样需要一个小的辅助函数，用于从 `type` 字符串（如 `'TIMESTAMP(6)'`）中解析出小数秒的精度值。

#### 3.2.3. `CLI` 层 - 用户意图翻译

- **涉及模块**: `cli/commands/schema.py`
- **涉及模型**: `_decompose_single_table_schema` 函数。
- **逻辑描述**: 修改此函数，在遍历用户定义的 `rules` 数组时，增加对 `length`, `precision`, `scale`, `datetime_precision` 这几个新可选键的检查。如果用户在规则中定义了这些键，则将它们及其值一并添加到为 `SCHEMA` 规则构建的 `columns_map` 参数字典中。

#### 3.2.4. `Rule Registry` 层 - 规则合法化

- **涉及模块**: `core/registry/builtin_rule_types.py`
- **涉及模型**: `SCHEMA` 规则的 `parameters_schema` 定义。
- **逻辑描述**: 更新 `SCHEMA` 规则的参数JSON Schema。在 `columns` 的 `additionalProperties` 中，将 `length`, `precision`, `scale`, `datetime_precision` 添加为可选的 `integer` 类型属性。由于它们不是必需的，这保证了向后兼容性。
