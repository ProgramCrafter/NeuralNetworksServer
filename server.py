import traceback
import sys
import os

from main import Predictor

net = Predictor()

def load_file(path):
  with open(os.path.abspath(__file__ + '/../' + path)) as f:
    return f.read()

def application(environ, start_response):
  response_headers = [
    ('Content-Type', 'text/html'),
  ]

  if environ.get('PATH_INFO') == '/':
    status = '200 OK'
    content = load_file('index.html')

    if environ['QUERY_STRING']:
      try:
        query = environ['QUERY_STRING'].split('&')
        data = [0, 0, '']

        for q in query:
          if q.startswith('time='):
            data[0] = int(q.removeprefix('time='))
          elif q.startswith('memo='):
            data[1] = int(q.removeprefix('memo='))
          elif q.startswith('lang='):
            data[2] = q.removeprefix('lang=').replace('+', ' ').replace('%2B', '+')

        rating = net.predict(*data)

        content = (content
          .replace('HiDdEn', 'visible')
          .replace('%TIME%', str(data[0]))
          .replace('%MEMORY%', str(data[1]))
          .replace('%LANG%', data[2])
          .replace('%RATING%', '%.1f' % rating))
      except:
        content += '<div id="error">%s</div>' % traceback.format_exc()

  else:
    status = '302 Moved Temporarily'

    qs = '?' + environ['QUERY_STRING'] if 'QUERY_STRING' in environ else ''

    response_headers.append(('Location', '/' + qs))
    content = ''

  content += '<!-- %s -->' % str(environ)

  response_headers.append(('Content-Length', str(len(content))))

  start_response(status, response_headers)
  yield content.encode('utf-8')

print('[Ratingers|NeuralNetworks] started server', file=sys.stdout, flush=True)
