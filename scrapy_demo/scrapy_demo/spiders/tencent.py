# -*- coding: utf-8 -*-
import scrapy
from scrapy_demo.items import ScrapyDemoItem


class TencentSpider(scrapy.Spider):
    name = 'tencent'
    allowed_domains = ['tencent.com']
    baseURL = "http://hr.tencent.com/position.php?&start="
    # 通过分析HTML中地址
    offset = 0
    start_urls = [baseURL + str(offset)]

    def parse(self, response):
        node_list = response.xpath("//tr[@class = 'even'] | //tr[@class='odd']")
        # td[2]
        for node in node_list:
            item = ScrapyDemoItem()  # 引入字段类
            item['positionName'] = node.xpath("./td[1]/a/text()").extract()[0].encode('utf-8')
            item['positionLink'] = node.xpath("./td[1]/a/@href").extract()[0].encode("utf-8")  # 链接属性
            item['positionType'] = node.xpath("./td[2]/text()").extract()[0].encode("utf-8")
            item['positionNumber'] = node.xpath("./td[3]/text()").extract()[0].encode("utf-8")
            item['workLocation'] = node.xpath("./td[4]/text()").extract()[0].encode("utf-8")
            item['publishTime'] = node.xpath("./td[5]/text()").extract()[0].encode("utf-8")
            # 返回给管道处理
            yield item
# 首先固定20
        if self.offset < 20:
            self.offset += 10
            url = self.baseURL + str(self.offset)
            yield scrapy.Request(url, callback=self.parse)
        # pass
