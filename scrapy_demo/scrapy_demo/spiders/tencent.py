# -*- coding: utf-8 -*-
import scrapy
from ..items import ScrapyDemoItem


class TencentSpider(scrapy.Spider):
    name = 'tencent'
    allowed_domains = ['tencent.com']
    start_urls = ['http://tencent.com/']

    def parse(self, response):
        pass
