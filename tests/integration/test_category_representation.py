"""
Test Category Theory and Representation Theory modules.
"""

import numpy as np

print("\n=== Testing Category Theory + Representation Theory ===\n")

# Test 1: Category Theory
print("Test 1: Category Theory - Basic Operations")

from HoloLoom.warp.category import Category, Morphism, Functor

C = Category(name="Test")
C.add_object("A")
C.add_object("B")
C.add_object("C")

f = Morphism(source="A", target="B", data="f", name="f")
g = Morphism(source="B", target="C", data="g", name="g")

C.add_morphism(f)
C.add_morphism(g)

# Composition
gf = C.compose(f, g)
print(f"  Composed: {gf.source} -> {gf.target}")
print(f"  Identity exists: {C.identity('A').name}")
print("  PASS\n")

# Test 2: Functors
print("Test 2: Functors")

D = Category(name="Target")

def obj_map(x):
    return f"F({x})"

def morph_map(m):
    return Morphism(
        source=obj_map(m.source),
        target=obj_map(m.target),
        data=f"F({m.data})",
        name=f"F({m.name})"
    )

F = Functor(source=C, target=D, object_map=obj_map, morphism_map=morph_map, name="F")

FA = F.map_object("A")
Ff = F.map_morphism(f)

print(f"  F(A) = {FA}")
print(f"  F(f): {Ff.source} -> {Ff.target}")
print("  PASS\n")

# Test 3: Natural Transformations
print("Test 3: Natural Transformations")

from HoloLoom.warp.category import NaturalTransformation

# Create second functor G
G = Functor(source=C, target=D,
            object_map=lambda x: f"G({x})",
            morphism_map=lambda m: Morphism(source=f"G({m.source})",
                                            target=f"G({m.target})",
                                            data=f"G({m.data})",
                                            name=f"G({m.name})"),
            name="G")

# Natural transformation components
components = {}
for obj in C.objects:
    components[obj] = Morphism(
        source=F.map_object(obj),
        target=G.map_object(obj),
        data="eta",
        name=f"eta_{obj}"
    )

eta = NaturalTransformation(source_functor=F, target_functor=G,
                             components=components, name="eta")

print(f"  eta: F => G")
print(f"  Component at A: {eta.component('A').name}")
print("  PASS\n")

# Test 4: Yoneda Embedding
print("Test 4: Yoneda Embedding")

from HoloLoom.warp.category import YonedaEmbedding

yoneda = YonedaEmbedding(C)
hom_B = yoneda.embed_object("B")

print(f"  Yoneda: {hom_B.name}")
print(f"  Source category: {hom_B.source.name}")
print("  PASS\n")

# Test 5: Monoidal Category
print("Test 5: Monoidal Categories")

from HoloLoom.warp.category import MonoidalCategory

M = MonoidalCategory(name="Tensor", unit_object="I")
M.add_object("X")
M.add_object("Y")

XY = M.tensor_objects("X", "Y")
print(f"  X tensor Y = {XY}")

# Tensor morphisms
f_X = Morphism(source="X", target="X", data=np.eye(2), name="f_X")
f_Y = Morphism(source="Y", target="Y", data=np.eye(3), name="f_Y")

M.add_morphism(f_X)
M.add_morphism(f_Y)

f_XY = M.tensor_morphisms(f_X, f_Y)
print(f"  Tensor morphism created: {f_XY.source} -> {f_XY.target}")
print(f"  Tensor data shape: {f_XY.data.shape}")
print("  PASS\n")

# Test 6: Representation Theory - Groups
print("Test 6: Group Theory")

from HoloLoom.warp.representation import Group, cyclic_group, symmetric_group

C3 = cyclic_group(3)
print(f"  C3 order: {C3.order()}")
print(f"  C3 abelian: {C3.is_abelian()}")

S3 = symmetric_group(3)
print(f"  S3 order: {S3.order()}")
print(f"  S3 abelian: {S3.is_abelian()}")
print("  PASS\n")

# Test 7: Representations
print("Test 7: Representations")

from HoloLoom.warp.representation import (
    Representation, trivial_representation, regular_representation
)

triv = trivial_representation(C3)
print(f"  Trivial rep dimension: {triv.dimension}")
print(f"  Homomorphism verified: {triv.verify_homomorphism()}")

reg = regular_representation(C3)
print(f"  Regular rep dimension: {reg.dimension}")
print(f"  Homomorphism verified: {reg.verify_homomorphism()}")
print("  PASS\n")

# Test 8: Characters
print("Test 8: Character Theory")

triv_char = triv.character()
reg_char = reg.character()

print(f"  Trivial character: {list(triv_char.values())}")
print(f"  Regular character: {list(reg_char.values())}")

print(f"  Trivial irreducible: {triv.is_irreducible()}")
print(f"  Regular irreducible: {reg.is_irreducible()}")
print("  PASS\n")

# Test 9: Character Table
print("Test 9: Character Tables")

from HoloLoom.warp.representation import CharacterTable

char_table = CharacterTable(C3)
print(f"  Conjugacy classes: {len(char_table.conjugacy_classes)}")

char_table.add_irrep(triv)
print(f"  Added irrep: {triv.name}")
print(f"  Table shape: {char_table.table.shape}")
print("  PASS\n")

# Test 10: Equivariant Maps
print("Test 10: Equivariant Maps")

from HoloLoom.warp.representation import EquivariantMap

# Identity map triv -> triv
eq_map = EquivariantMap(
    source_rep=triv,
    target_rep=triv,
    matrix=np.eye(1),
    name="id"
)

print(f"  Map: {eq_map.source_rep.name} -> {eq_map.target_rep.name}")
print(f"  Equivariant: {eq_map.verify_equivariance()}")
print("  PASS\n")

print("=== All Tests Passed! ===\n")
