import pandas as pd
with open('keththarama.csv', 'rb') as f:
    text = f.read()
print('First 100 chars:', text[:100])
print('Last 100 chars:', text[-100:])
if b'\r\n' in text:
    print('Contains CRLF')
elif b'\r' in text:
    print('Contains CR')
elif b'\n' in text:
    print('Contains LF')
else:
    print('No standard newlines')
