import yaml

# Read the environment.yml file
with open("environment.yml", "r") as f:
    env_data = yaml.load(f, Loader=yaml.FullLoader)

# Remove version constraints from each dependency
cleaned_dependencies = []

for dep in env_data.get("dependencies", []):
    if isinstance(dep, str):  # It's a direct conda dependency
        dep = dep.split("=")[0]
        cleaned_dependencies.append(dep)
    elif isinstance(dep, dict) and "pip" in dep:
        cleaned_pip_deps = []
        for pip_dep in dep["pip"]:
            pip_dep = pip_dep.split("==")[0]  # pip usually uses '==' for version specification
            cleaned_pip_deps.append(pip_dep)
        dep["pip"] = cleaned_pip_deps
        cleaned_dependencies.append(dep)

env_data["dependencies"] = cleaned_dependencies

# Write the cleaned data back to the file
with open("environment_cleaned.yml", "w") as f:
    yaml.dump(env_data, f)

print("Cleaned environment file saved as environment_cleaned.yml")
