import os

print("Running isort...")
os.system("uv run isort sbq_python_template tests")
print("---------------\n\n")

print("Running pytest...")
os.system("uv run pytest sbq_python_template tests")
print("---------------\n\n")

print("Running mypy...")
os.system("uv run mypy sbq_python_template tests")
print("---------------\n\n")

print("Running pylint...")
os.system("uv run pylint sbq_python_template tests")
print("---------------\n\n")
