"""
Reads xml synapse recipe and outputs LaTex table (rows) with parameters
last modified: András Ecker 05.2021
"""

# see FUNCZ-183: pip install --index-url https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ fz-td-recipe
from fz_td_recipe import Recipe


def recipe_to_tex(recipe, f_name):
    """Reads synapse classes from xml and formats them to be used in a LaTex table"""
    with open(f_name, "w") as f:
        for s in recipe.synapse_properties.classes:
            tex = r"%s & %.1f$\pm$%.1f & %.2f$\pm$%.2f & %i$\pm$%i & %i$\pm$%i & %.1f & %.2f$\pm$%.2f & %.1f & %.2f \\" % (s.id,
                    getattr(s, "gsyn"), getattr(s, "gsynSD"), getattr(s, "u"), getattr(s, "uSD"),
                    getattr(s, "d"), getattr(s, "dSD"), getattr(s, "f"), getattr(s, "fSD"), getattr(s, "nrrp"),
                    getattr(s, "dtc"), getattr(s, "dtcSD"), getattr(s, "gsynSRSF"), getattr(s, "uHillCoefficient"))
            f.write("%s\n" % tex)


if __name__ == "__main__":
    xmlf_name = "/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/bioname/builderRecipeAllPathways.xml"
    outf_name = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/recipe.tex"
    recipe_to_tex(Recipe(xmlf_name), outf_name)

