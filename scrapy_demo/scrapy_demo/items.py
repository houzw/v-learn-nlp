# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapyDemoItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    positionName = scrapy.Field()
    positionLink = scrapy.Field()
    # 职位类型
    positionType = scrapy.Field()
    # 职位人数
    positionNumber = scrapy.Field()
    # 工作地点
    workLocation = scrapy.Field()
    # 发布时点
    publishTime = scrapy.Field()
    pass
