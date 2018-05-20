>https://segmentfault.com/a/1190000012041391

1. 新建项目（scrapy startproject xxx）:新建一个新的爬虫项目
2. 明确目标（编写items.py）:明确你想要抓取的目标
3. 制作爬虫（spiders/xx_spider.py）:制作爬虫开始爬取网页
4. 存储内容（pipelines.py）:设计管道存储爬取内容


1.创建项目 scrapy project XXX
2.scarpy genspider xxx "http://www.xxx.com"
3.编写 items.py, 明确需要提取的数据
4.编写 spiders/xxx.py, 编写爬虫文件，处理请求和响应，以及提取数据（yield item）
5.编写 pipelines.py, 编写管道文件，处理spider返回item数据,比如本地数据持久化，写文件或存到表中。
6.编写 settings.py，启动管道组件ITEM_PIPELINES，以及其他相关设置
7.执行爬虫 scrapy crawl xxx