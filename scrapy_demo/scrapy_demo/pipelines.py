# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import json


# 在编码函数之前写一个编码类，只要检查到了是bytes类型的数据就把它转化成str类型。
# https://blog.csdn.net/bear_sun/article/details/79397155
# https://stackoverflow.com/questions/45962216/convert-a-dict-of-bytes-to-json
# 中文的话需要
class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            # str(o, encoding='gbk') 不行
            return str(o, encoding='utf-8')
        return json.JSONEncoder.default(self, o)


import codecs

class ScrapyDemoPipeline(object):
    def __init__(self):
        self.f = codecs.open('tencent.json', 'w', encoding='utf-8')

    # 所有item使用的共同的管道
    def process_item(self, item, spider):
        # json模块不支持bytes类型，要先将bytes转换为str格式
        content = json.dumps(dict(item), ensure_ascii=False, cls=JsonEncoder) + ",\n"
        self.f.write(content)
        return item

    def close_spider(self, spider):
        self.f.close()
