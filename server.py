import traceback
import sys
import os

try:
  from network import Predictor

  net = Predictor()
except:
  traceback.print_exc()

def load_file(path):
  with open(os.path.abspath(__file__ + '/../' + path)) as f:
    return f.read()

expected_location = '/cf-task-rating'

def application(environ, start_response):
  response_headers = [
    ('Content-Type', 'text/html'),
  ]

  if environ['PATH_INFO'] == expected_location:
    status = '200 OK'
        
    version = ('v=1' in environ['QUERY_STRING'].split('&')) + 0
    
    if 'ru' in environ.get('HTTP_ACCEPT_LANGUAGE', ''):
      content = load_file('index-v%d.html' % version)
    else:
      content = load_file('index-en-v%d.html' % version)

    if environ['QUERY_STRING']:
      try:
        query = environ['QUERY_STRING'].split('&')
        
        if version == 0:
          data = [0, 0, '']
          
          for q in query:
            if q.startswith('time='):
              data[0] = int(q.removeprefix('time='))
            elif q.startswith('memo='):
              data[1] = int(q.removeprefix('memo='))
            elif q.startswith('lang='):
              data[2] = q.removeprefix('lang=').replace('+', ' ').replace('%2B', '+')
          
          rating = net.predict(*data)
        else:
          data = [0, 0, '', 0]
          
          for q in query:
            if q.startswith('time='):
              data[0] = int(q.removeprefix('time='))
            elif q.startswith('memo='):
              data[1] = int(q.removeprefix('memo='))
            elif q.startswith('lang='):
              data[2] = q.removeprefix('lang=').replace('+', ' ').replace('%2B', '+')
            elif q.startswith('subtime='):
              data[3] = int(q.removeprefix('subtime='))
          
          rating = net.predict_v1(*data)

        content = (content
          .replace('HiDdEn', 'visible')
          .replace('%TIME%', str(data[0]))
          .replace('%MEMORY%', str(data[1]))
          .replace('%LANG%', data[2])
          .replace('%RATING%', '%.1f' % rating))
      except:
        content = content.replace('<!--ERROR PLACEHOLDER-->',
          '<div id="error">%s</div>' % traceback.format_exc().replace('\n', '<br>'))

  else:
    status = '302 Moved Temporarily'

    qs = '?' + environ['QUERY_STRING'] if environ['QUERY_STRING'] else ''

    response_headers.append(('Location', expected_location + qs))
    content = ''

  response_headers.append(('Content-Length', str(len(content))))

  start_response(status, response_headers)
  yield content.encode('utf-8')

print('[Ratingers|NeuralNetworks] started server', file=sys.stdout, flush=True)
