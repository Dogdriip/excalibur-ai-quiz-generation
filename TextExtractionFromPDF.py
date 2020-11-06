from tika import parser
import re

raw = parser.from_file('./sample.pdf')
content = raw['content']


content = content.strip()
content = re.sub(
    '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', content)
content = ' '.join(content.split())

print(content)
