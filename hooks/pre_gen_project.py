import sys
from cookiecutter.main import cookiecutter

service_name = "{{ cookiecutter.service_name }}"
service_name = service.replace("_", "-")
service_name = service.replace(" ", "-")

package_name = service_name.replace("-", "_")

cookiecutter(
    "cookiecutter-django",
    extra_context={"package_name": package_name}
)

print (1)
print (service_name, package_name)

print (2)
print ("{{ cookiecutter.service_name }}", "{{ cookiecutter.package_name }}")

sys.exit()