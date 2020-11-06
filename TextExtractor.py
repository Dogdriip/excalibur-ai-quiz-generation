from tika import parser
import re


class TextExtractor(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def cleanup_text(self, content):
        content = content.strip()
        content = re.sub(
            r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', content)
        content = ' '.join(content.split())
        return content

    def extract_text(self):
        raw = parser.from_file(self.filepath)
        content = raw['content']
        content = self.cleanup_text(content)
        return content
