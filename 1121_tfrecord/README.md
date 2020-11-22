# Outline

1. 正常怎么存成 字符串/怎么解压字符串 + 对文件进行写入
2. 对文件进行读出
3. Youtube8m 使用yield进行数据生成
4. Youtube8m 使用reader 进行数据读入



# Youtube-8m

要读取的数据集

```
# 1 example 1 video
features: {
  feature: {
    key  : "id"
    value: {
      bytes_list: {
        value: (Video id)
      }
    }
  }
  feature: {
    key  : "labels"
    value: {
      int64_list: {
        value: [1, 522, 11, 172]  # label list
      }
    }
  }
  feature: {
    # Average of all 'rgb' features for the video
    key  : "mean_rgb"
    value: {
      float_list: {
        value: [1024 float features]
      }
    }
  }
  feature: {
    # Average of all 'audio' features for the video
    key  : "mean_audio"
    value: {
      float_list: {
        value: [128 float features]
      }
    }
  }
}

```



