import os

print('Running isort...')
os.system("poetry run isort sbq_python_template tests")
print('---------------\n\n')

print('Running pytest...')
os.system("poetry run pytest sbq_python_template tests")
print('---------------\n\n')

print('Running mypy...')
os.system("poetry run mypy sbq_python_template tests")
print('---------------\n\n')

print('Running pylint...')
os.system("poetry run pylint sbq_python_template tests")
print('---------------\n\n')
