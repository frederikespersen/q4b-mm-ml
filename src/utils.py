"""
    MM-ML Analysis Utils
    --------------------------------------------------------------------------------

    Utils for analysing the difference between conventional (MM) molecular dynamics-
    generated ensembles and ensembles corrected with non-equilibrium resampling
    using a machine learning potential (ML) of a bound/unbound ligand.

    F. E. Knudsen

    --------------------------------------------------------------------------------
"""

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.analysis import distances as Distances
import MDAnalysis.transformations as trans

from rdkit import Chem as rdc
from rdkit.Chem import Draw as rdd

from matplotlib import pyplot as plt
from matplotlib import ticker as tck
from matplotlib.patches import Rectangle
import seaborn as sns

import numpy as np
import pandas as pd

from scipy.spatial import distance

from tqdm import tqdm



# ················································································· #
# ································ P L O T T I N G ································ #
# ················································································· #

# Standard color palette
palette = {
    'red':    '#dc706b',
    'mint':   '#85c3c4',
    'orange': '#eeb259',
    'indigo': '#6e74af',
    'blue':   '#6591c8',
    'yellow': '#f8d351',
    'green':  '#73af60',
    'purple': '#b094bf'
}



# ················································································· #
# ······························· S E L E C T I O N ······························· #
# ················································································· #

def print_atom_names(universe: mda.Universe,
                     selection_query: str,
                     save_to: str = False):

    # Setting bonds for residues
    if 'protein' in selection_query or 'resid' in selection_query:
        universe.select_atoms(selection_query).guess_bonds()
    
    # Initialising rdkit molecule
    rdk_mol = rdc.RWMol()

    # Add atoms to the RDKit molecule, and save atom name mapping
    atom_mapping = {}
    select_atoms = universe.select_atoms(selection_query)
    for atom in select_atoms:
        rdk_atom = rdc.Atom(atom.element)
        rdk_idx = rdk_mol.AddAtom(rdk_atom)
        atom_mapping[atom.index] = rdk_idx

    # Add bonds the RDKit molecule
    for bond in select_atoms.bonds:
        rdk_mol.AddBond(atom_mapping[bond[0].index],
                        atom_mapping[bond[1].index],
                        rdc.BondType.SINGLE)  # Drawing single bonds for ease

    # Convert to a regular RDKit molecule
    rdk_mol = rdk_mol.GetMol()

    # Add atom indices to atom labels
    for rdk_atom in rdk_mol.GetAtoms():
        rdk_atom.SetProp('atomLabel', select_atoms[rdk_atom.GetIdx()].name)

    # Draw and optionally save the molecule with atom labels
    img = rdd.MolToImage(rdk_mol)
    if save_to:
        img.save(save_to)
    return img

# ················································································· #

def ligand_atom_name_map(ligand_query: str,
                         from_universe: mda.Universe,
                         to_universe: mda.Universe
                         ) -> dict:
    # Making mapping from complex atom names to solvent atom names
    from_ligand_atoms = from_universe.select_atoms(ligand_query)
    to_ligand_atoms = to_universe.select_atoms(ligand_query)
    atom_name_map = {}
    for (fla, tla) in zip(from_ligand_atoms, to_ligand_atoms):
        atom_name_map[fla.name] = tla.name
    return atom_name_map

# ················································································· #

def ligand_selection_query(ligand_query: str,
                           atom_names,
                           atom_name_map: dict[str] = False
                           ) -> str:
    if atom_name_map:
        atom_names = (atom_name_map[an] for an in atom_names)
    return ligand_query + ' and name ' + ' '.join(atom_names)


# ················································································· #
# ······························· D I H E D R A L S ······························· #
# ················································································· #

def dihedral_time_series(selection: mda.AtomGroup):

    dihedrals = Dihedral([selection]).run().results.angles.T

    return dihedrals

# ················································································· #

def plot_ligand_dihedral_distributions(ligand_query: str,
                                       atom_names: tuple[str, str, str, str],
                                       mm_complex: mda.Universe,
                                       mm_solvent: mda.Universe,
                                       ml_complex: mda.Universe,
                                       ml_solvent: mda.Universe,
                                       atom_name_scheme: str = 'complex',
                                       binsize: int = 5,
                                       size: tuple[float, float] = (3.5, 2),
                                       alpha: float = 0.6,
                                       save_to: str = False):
    # Assembling queries
    if atom_name_scheme == 'solvent':
        s2c_atom_name_map = ligand_atom_name_map(ligand_query, mm_solvent, mm_complex)
        complex_ligand_selection_query = ligand_selection_query(ligand_query, atom_names, s2c_atom_name_map)
        solvent_ligand_selection_query = ligand_selection_query(ligand_query, atom_names)
    elif atom_name_scheme == 'complex':
        c2s_atom_name_map = ligand_atom_name_map(ligand_query, mm_complex, mm_solvent)
        complex_ligand_selection_query = ligand_selection_query(ligand_query, atom_names)
        solvent_ligand_selection_query = ligand_selection_query(ligand_query, atom_names, c2s_atom_name_map)
    else:
        raise KeyError("Must specify either 'solvent' or 'complex' for ligand atom naming scheme.")

    # Selecting atoms
    atoms_mm_complex = mm_complex.select_atoms(complex_ligand_selection_query)
    atoms_mm_solvent = mm_solvent.select_atoms(solvent_ligand_selection_query)
    atoms_ml_complex = ml_complex.select_atoms(complex_ligand_selection_query)
    atoms_ml_solvent = ml_solvent.select_atoms(solvent_ligand_selection_query)

    # Calculating dihedral angles
    dihedrals_mm_complex = dihedral_time_series(atoms_mm_complex)
    dihedrals_mm_solvent = dihedral_time_series(atoms_mm_solvent)
    dihedrals_ml_complex = dihedral_time_series(atoms_ml_complex)
    dihedrals_ml_solvent = dihedral_time_series(atoms_ml_solvent)

    # Setting bins
    bins = np.linspace(-180, 180, int(360 / binsize + 1))
    bins_mid = (bins[1:] + bins[:-1]) / 2

    # Calculating densities
    density_mm_complex = np.histogram(dihedrals_mm_complex, bins=bins, density=True)[0]
    density_mm_solvent = np.histogram(dihedrals_mm_solvent, bins=bins, density=True)[0]
    density_ml_complex = np.histogram(dihedrals_ml_complex, bins=bins, density=True)[0]
    density_ml_solvent = np.histogram(dihedrals_ml_solvent, bins=bins, density=True)[0]

    # Initializing figure
    fig = plt.figure(figsize=size)
    ax = plt.gca()

    # Setting axis limits
    plt.xlim(-180, 180)
    ymax = max([
        max(density_mm_complex),
        max(density_mm_solvent),
        max(density_ml_complex),
        max(density_ml_solvent)]
    ) * 1.05
    plt.ylim(-ymax, ymax)

    # Complex-solvent interface
    plt.hlines(0, xmin=-180, xmax=180,
               linewidth=0.5, color='black')
    ax.add_patch(Rectangle((-180, 0), 360, -ymax,
                           fill=False, color='grey', hatch='..', alpha=0.3))
    plt.text(185, ymax, "complex",
             rotation=-90, fontsize=10, va='top')
    plt.text(185, -ymax, "solvent",
             rotation=-90, fontsize=10, va='bottom')

    # Cover up patterned background patch up with full alpha white histograms
    plt.bar(bins_mid, density_mm_complex,
            alpha=1, color='white', width=binsize)
    plt.bar(bins_mid, density_ml_complex,
            alpha=1, color='white', width=binsize)
    plt.bar(bins_mid, -density_mm_solvent,
            alpha=1, color='white', width=binsize)
    plt.bar(bins_mid, -density_ml_solvent,
            alpha=1, color='white', width=binsize)

    # Histograms; Complex on top, solvent on bottom; MM in red, ML in blue
    plt.bar(bins_mid, density_mm_complex,
            alpha=alpha, color=palette['red'], width=binsize)
    plt.bar(bins_mid, density_ml_complex,
            alpha=alpha, color=palette['mint'], width=binsize)
    plt.bar(bins_mid, -density_mm_solvent,
            alpha=alpha, color=palette['red'], width=binsize)
    plt.bar(bins_mid, -density_ml_solvent, alpha=alpha,
            color=palette['mint'], width=5)

    # Formatting axes
    ax.axes.get_xaxis().set_major_formatter(tck.FormatStrFormatter('%g°'))
    ax.axes.get_xaxis().set_major_locator(tck.MultipleLocator(base=180))
    ax.axes.get_yaxis().set_visible(False)

    plt.tight_layout()

    # Optionally saving
    if save_to:
        fig.savefig(save_to)

    # Returning distribution data
    return {'mm_complex': dihedrals_mm_complex, 'mm_solvent': dihedrals_mm_solvent,
            'ml_complex': dihedrals_ml_complex, 'ml_solvent': dihedrals_ml_solvent}



# ················································································· #
# ······························· D I S T A N C E S ······························· #
# ················································································· #

def distance_time_series(universe: mda.Universe,
                         selection1: mda.AtomGroup,
                         selection2: mda.AtomGroup,
                         precalc_file: str = False):

    # Try to load precalculated distances
    try:
        if not precalc_file:
            raise FileNotFoundError
        distances = np.load(precalc_file)
        print(f"Loaded distances from file: {precalc_file}")

    # Calculating from scratch
    except FileNotFoundError:

        # Calculating distance time series
        distances = []
        for timestep in tqdm(universe.trajectory):
            dist = Distances.distance_array(selection1.positions,
                                            selection2.positions,
                                            box=universe.dimensions)
            distances.append(dist)
        distances = np.array(distances)
        if precalc_file:
            np.save(precalc_file, distances)
            print(f"Saved distances to file: {precalc_file}")

    return distances

# ················································································· #

def distance_com_time_series(universe: mda.Universe,
                             selection1: mda.AtomGroup,
                             selection2: mda.AtomGroup,
                             precalc_file: str = False):
    # Try to load precalculated distances
    try:
        if not precalc_file:
            raise FileNotFoundError
        distances = np.load(precalc_file)
        print(f"Loaded distances from file: {precalc_file}")

    # Calculating from scratch
    except FileNotFoundError:

        # Calculating distance time series
        distances = []
        for timestep in tqdm(universe.trajectory):
            dist = Distances.distance_array(selection1.center_of_mass(),
                                            selection2.center_of_mass(),
                                            box=universe.dimensions)
            distances.append(dist)
        distances = np.array(distances)[:, 0, 0]
        if precalc_file:
            np.save(precalc_file, distances)
            print(f"Saved distances to file: {precalc_file}")

    return distances

# ················································································· #

def plot_distance_matrix(universe: mda.Universe,
                         selection_query1: str,
                         selection_query2: str,
                         sel1_cutoff: float = False,
                         sel2_cutoff: float = False,
                         measure: str = 'mean',
                         precalc_file: str = False,
                         save_to: str = False):
    # Making atom selections
    sel1 = universe.select_atoms(selection_query1)
    sel2 = universe.select_atoms(selection_query2)

    # Calculating distances
    distances = distance_time_series(universe, sel1, sel2, precalc_file)

    # Setting atom labels
    sel1_labels = [
        f"{atom.resname}{str(atom.resid)} (${atom.name[0]}{'_{' + atom.name[1:] + '}' if len(atom.name) > 1 else ''}$)"
        for atom in sel1]
    sel2_labels = [
        f"{atom.resname}{str(atom.resid)} (${atom.name[0]}{'_{' + atom.name[1:] + '}' if len(atom.name) > 1 else ''}$)"
        for atom in sel2]

    # Aggregating time axis
    if measure == 'mean':
        distance_matrix = pd.DataFrame(distances.mean(axis=0), index=sel1_labels, columns=sel2_labels).T
    elif measure == 'median':
        distance_matrix = pd.DataFrame(distances.mean(axis=0), index=sel1_labels, columns=sel2_labels).T
    else:
        raise ValueError("Must choose either 'mean' or 'median' measure.")

    # Filtering by cutoff
    if sel1_cutoff:
        distance_matrix = distance_matrix.loc[(distance_matrix < sel1_cutoff).any(axis=0)]
    if sel2_cutoff:
        distance_matrix = distance_matrix.loc[(distance_matrix < sel2_cutoff).any(axis=1)]
    fig = sns.clustermap(distance_matrix, cmap="flare", vmin=2, vmax=max([sel1_cutoff, sel2_cutoff, 5]) * 2)

    # Optionally saving
    if save_to:
        fig.savefig(save_to)

    return distance_matrix

# ················································································· #

def plot_com_distance_distributions(mm_complex: mda.Universe,
                                    ml_complex: mda.Universe,
                                    selection_query1: str,
                                    selection_query2: str,
                                    binsize: float = 0.2,
                                    xlims: tuple[float, float] = (2,10),
                                    size: tuple[float, float] = (3.0, 1.5),
                                    alpha: float = 0.6,
                                    save_to: str = False):
    # Making atom selections
    mm_sel1 = mm_complex.select_atoms(selection_query1)
    mm_sel2 = mm_complex.select_atoms(selection_query2)
    ml_sel1 = ml_complex.select_atoms(selection_query1)
    ml_sel2 = ml_complex.select_atoms(selection_query2)

    # Calculating distances
    distances_mm = distance_com_time_series(mm_complex, mm_sel1, mm_sel2)
    distances_ml = distance_com_time_series(ml_complex, ml_sel1, ml_sel2)

    # Setting bins
    maxbin = np.ceil(max([max(distances_mm), max(distances_ml)]))
    bins = np.linspace(0, maxbin, int(maxbin / binsize + 1))
    bins_mid = (bins[1:] + bins[:-1]) / 2

    # Initializing figure
    fig = plt.figure(figsize=size)
    ax = plt.gca()

    # Histograms
    plt.hist(distances_mm, bins=bins, density=True,
             alpha=alpha, label="MM", color=palette['red'])
    plt.hist(distances_ml, bins=bins, density=True,
             alpha=alpha, label="ML", color=palette['mint'])

    # Setting axis limits
    midpoint = (np.median(distances_mm) + np.median(distances_ml)) / 2
    plt.xlim(*xlims)

    # Formatting axes
    plt.gca().axes.get_xaxis().set_major_formatter(tck.FormatStrFormatter('%g Å'))
    plt.gca().axes.get_xaxis().set_major_locator(tck.MultipleLocator(base=2))
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    
    # Optionally saving
    if save_to:
        fig.savefig(save_to)

    return {'mm_complex': distances_mm, 'ml_complex': distances_ml}

# ················································································· #

def plot_single_atom_distance_distributions(mm_complex: mda.Universe,
                                            ml_complex: mda.Universe,
                                            selection_query1: str,
                                            selection_query2: str,
                                            binsize: float = 0.2,
                                            xwidth: float = 2.5,
                                            alpha: float = 0.6,
                                            save_to: str = False):
    # Making atom selections
    mm_sel1 = mm_complex.select_atoms(selection_query1)
    mm_sel2 = mm_complex.select_atoms(selection_query2)
    ml_sel1 = ml_complex.select_atoms(selection_query1)
    ml_sel2 = ml_complex.select_atoms(selection_query2)

    # Calculating distances
    distances_mm = distance_time_series(mm_complex, mm_sel1, mm_sel2)
    distances_ml = distance_time_series(ml_complex, ml_sel1, ml_sel2)

    # Setting bins
    maxbin = np.ceil(max([distances_mm.max(), distances_ml.max()]))
    bins = np.linspace(0, maxbin, int(maxbin / binsize + 1))
    bins_mid = (bins[1:] + bins[:-1]) / 2

    # Initializing figure
    fig, axes = plt.subplots(nrows=len(mm_sel1),
                             ncols=len(mm_sel2),
                             figsize=(len(mm_sel1) * 1.5, len(mm_sel1) * 3))

    # Reshaping axes array to make it iterable
    if len(mm_sel2) == 1:
        axes = axes[:,np.newaxis]
    elif len(mm_sel1) == 1:
        axes = axes[np.newaxis,:]

    for i, atom1 in enumerate(mm_sel1):
        for j, atom2 in enumerate(mm_sel2):
            # Histograms
            axes[i, j].set_title(f"{atom1.resname}-{atom1.name} and {atom2.resname}-{atom2.name}")
            axes[i, j].hist(distances_mm[:, i, j], bins=bins, density=True,
                            alpha=alpha, label="MM", color=palette['red'])
            axes[i, j].hist(distances_ml[:, i, j], bins=bins, density=True,
                            alpha=alpha, label="ML", color=palette['mint'])

            # Setting axis limits
            axes[i, j].set_xlim(bins.min(), bins.max())

            # Formatting axes
            axes[i, j].axes.get_xaxis().set_major_formatter(tck.FormatStrFormatter('%g Å'))
            axes[i, j].axes.get_xaxis().set_major_locator(tck.MultipleLocator(base = 1 if bins.max()-bins.min() < 8 else 2))
            axes[i, j].axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.show()

    # Optionally saving
    if save_to:
        fig.savefig(save_to)

    return {'mm_complex': (distances_mm, mm_sel1, mm_sel2), 'ml_complex': (distances_ml, ml_sel1, ml_sel2)}

# ················································································· #

def plot_residue_substructure_com_distance_distributions(mm_complex: mda.Universe,
                                                         ml_complex: mda.Universe,
                                                         residue_query: str,
                                                         ligand_query: str,
                                                         hydrogen=False,
                                                         binsize: float = 0.2,
                                                         xwidth: float = 2.5,
                                                         alpha: float = 0.6,
                                                         save_to: str = False):
    # Optionally including hydrogen
    if not hydrogen:
        residue_query + ' and not element H'

    # Asserting protein and only one residue
    assert len(mm_complex.select_atoms('protein and ' + residue_query)) > 0
    assert len(mm_complex.select_atoms(residue_query).residues) == 1

    # Determining residue
    residue = mm_complex.select_atoms(residue_query).residues[0]

    # Calculating distances for substructures
    distances_mm = {}
    distances_ml = {}
    for substructure_name, subquery in residue_substructure_queries[residue.resname].items():

        # Making atom selections
        mm_residue = mm_complex.select_atoms(residue_query).select_atoms(subquery)
        mm_ligand =  mm_complex.select_atoms(ligand_query)
        ml_residue = ml_complex.select_atoms(residue_query).select_atoms(subquery)
        ml_ligand =  ml_complex.select_atoms(ligand_query)

        # Calculating distances
        distances_mm[substructure_name] = distance_com_time_series(mm_complex, mm_residue, mm_ligand)
        distances_ml[substructure_name] = distance_com_time_series(ml_complex, ml_residue, ml_ligand)

    # Setting bins
    maxbin = np.ceil(max([dist.max() for dist in distances_mm.values()] + [dist.max() for dist in distances_mm.values()]))
    bins = np.linspace(0, maxbin, int(maxbin / binsize + 1))
    bins_mid = (bins[1:] + bins[:-1]) / 2

    # Initializing figure
    fig, axes = plt.subplots(ncols=len(distances_mm),
                             figsize=(3*len(distances_mm), 1.5))

    for i, substructure_name in enumerate(residue_substructure_queries[residue.resname].keys()):
        # Histograms
        axes[i].set_title(f'{residue.resname.title()}{residue.resid} {substructure_name}')
        axes[i].hist(distances_mm[substructure_name], bins=bins, density=True,
                     alpha=alpha, label="MM", color=palette['red'])
        axes[i].hist(distances_ml[substructure_name], bins=bins, density=True,
                     alpha=alpha, label="ML", color=palette['mint'])

        # Setting axis limits
        axes[i].set_xlim(bins.min(), bins.max())

        # Formatting axes
        axes[i].axes.get_xaxis().set_major_formatter(tck.FormatStrFormatter('%g Å'))
        axes[i].axes.get_xaxis().set_major_locator(tck.MultipleLocator(base = 1 if bins.max()-bins.min() < 8 else 2))
        axes[i].axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.show()

    # Optionally saving
    if save_to:
        fig.savefig(save_to)

    return {'mm_complex': distances_mm, 'ml_complex': distances_ml}


# Substructure selection queries common for all amino acids
residue_basic_substructure_queries = {
    'All': 'all',
    'Backbone': 'backbone',
    'Sidechain': 'not backbone',
}

# Substructure selection queries specific individual amino acids
residue_substructure_queries = {
    'GLY': { **residue_basic_substructure_queries },
    'ALA': { **residue_basic_substructure_queries,
        'Methyl': 'name CB HB1 HB2 HB3'
    },
    'VAL': { **residue_basic_substructure_queries,
        'Methyl (1)': 'name CG1 HG11 HG12 HG13',
        'Methyl (2)': 'name CG2 HG21 HG22 HG23',
        'Isopropyl': 'name CB HB CG1 HG11 HG12 HG13 CG2 HG21 HG22 HG23'
    },
    'LEU': { **residue_basic_substructure_queries,
        'Methyl (1)': 'name CD1 HD11 HD12 HD13',
        'Methyl (2)': 'name CD2 HD21 HD22 HD23',
        'Isopropyl': 'name CG HG CD1 HD11 HD12 HD13 CD2 HD21 HD22 HD23',
        'Isobutyl': 'name CB HB2 HB3 CG HG CG HG CD1 HD11 HD12 HD13 CD2 HD21 HD22 HD23',
    },
    'ILE': { **residue_basic_substructure_queries,
        'Methyl (Delta)': 'name CG2 HG21 HG22 HG23',
        'Methyl (Gamma)': 'name CD1 HD11 HD12 HD13',
        'Ethyl': 'name CG1 HG12 HG13 CD1 HD11 HD12 HD13',
    },
    'PRO': { **residue_basic_substructure_queries,
        'Pyrrolidine': 'name CA CB CG CD N',
    },
    'SER': { **residue_basic_substructure_queries,
        'Hydroxyl': 'name OG HG',
    },
    'THR': { **residue_basic_substructure_queries,
        'Hydroxyl': 'name OG1 HG1',
        'Methyl': 'name CG2 HG21 HG22',
    },
    'CYS': { **residue_basic_substructure_queries,
        'Thiol': 'name SG HG',
    },
    'MET': { **residue_basic_substructure_queries,
        'Thioether': 'name SD',
        'Methyl': 'name CD HE1 HE2 HE3',
        'Methyl thioether': 'name SD CD HE1 HE2 HE3',
        'Linker': 'name CB HB2 HB3 CG HG2 HG3'
    },
    'ASN': { **residue_basic_substructure_queries,
        'Amide': 'name ND2 HD21 HD22 OD1',
    },
    'GLN': { **residue_basic_substructure_queries,
        'Amide': 'name NE2 HE21 HE22 OE1',
        'Linker': 'name CB HB2 HB3 CG HG2 HG3'
    },
    'ASP': { **residue_basic_substructure_queries,
        'Carboxyl': 'name CG OD1 OD2',
    },
    'GLU': { **residue_basic_substructure_queries,
        'Carboxyl': 'name CD OE1 OE2',
        'Linker': 'name CB HB2 HB3 CG HG2 HG3'
    },
    'LYS': { **residue_basic_substructure_queries,
        'Amino': 'name NZ HZ1 HZ2 HZ3',
        'Linker': 'name CB HB2 HB3 CG HG2 HG3 CD HD2 HD3 CE HE2 HE3'
    },
    'ARG': { **residue_basic_substructure_queries,
        'Guanidinyl': 'name NH1 HH11 HH12 NH2 HH21 HH22 NE HE CZ',
        'Linker': 'name CB HB2 HB3 CG HG2 HG3 CD HD2 HD3'
    },
    'HIS': { **residue_basic_substructure_queries,
        'Imidazole': 'name CG ND1 HD1 CD2 HD2 CE1 HE1 NE2 HE2',
    },
    'PHE': { **residue_basic_substructure_queries,
        'Phenyl': 'name CG CD1 HD1 CD2 HD2 CE1 HE1 CE2 HE2 CZ HZ',
        'Benzyl': 'name CB HB2 HB3 CG CD1 HD1 CD2 HD2 CE1 HE1 CE2 HE2 CZ HZ'
    },
    'TYR': { **residue_basic_substructure_queries,
        'Hydroxyl': 'name OH HH',
        'Phenyl': 'name CG CD1 HD1 CD2 HD2 CE1 HE1 CE2 HE2 CZ',
        'Benzyl': 'name CB HB2 HB3 CG CD1 HD1 CD2 HD2 CE1 HE1 CE2 HE2 CZ',
        'm-Cresoyl': 'name CB HB2 HB3 CG CD1 HD1 CD2 HD2 CE1 HE1 CE2 HE2 CZ OH HH',
    },
    'TRP': { **residue_basic_substructure_queries,
        'Indole': 'name CG CD1 HD1 CD2 NE1 HE1 CE2 CZ2 HZ2 CH2 HH2 CZ3 HZ3 CE3 HE3',
        'Benzene': 'name CD2 CE2 CZ2 HZ2 CH2 HH2 CZ3 HZ3 CE3 HE3',
        'Pyrrole': 'name CG CD1 HD1 CD2 NE1 HE1 CE2',
    }
}

# ················································································· #

def plot_residues_substructure_com_distance_divergence(mm_complex: mda.Universe,
                                                       ml_complex: mda.Universe,
                                                       residues_query: str,
                                                       ligand_query: str,
                                                       hydrogen=False,
                                                       scatter=False,
                                                       binsize: float = 0.2,
                                                       save_to: str = False,
                                                       save_scatter_to: str = False):

    # Optionally including hydrogen
    if not hydrogen:
        residues_query + ' and not element H'

    # Asserting protein
    assert len(mm_complex.select_atoms('protein and ' + residues_query)) > 0

    # Looping over residues
    jensen_shannon_distances = {}
    mean_diffs = {}
    for residue in mm_complex.select_atoms(residues_query).residues:

        # Selecting residue
        mm_complex_residue = mm_complex.select_atoms(residues_query).select_atoms(f'resid {residue.resid}')
        ml_complex_residue = ml_complex.select_atoms(residues_query).select_atoms(f'resid {residue.resid}')

        # Looping over substructures
        distances_mm = {}
        distances_ml = {}
        for substructure_name, subquery in residue_substructure_queries[residue.resname].items():

            # Making atom selections
            mm_residue_sub = mm_complex_residue.select_atoms(subquery)
            mm_ligand =  mm_complex.select_atoms(ligand_query)
            ml_residue_sub = ml_complex_residue.select_atoms(subquery)
            ml_ligand =  ml_complex.select_atoms(ligand_query)

            # Calculating distances
            distances_mm[substructure_name] = distance_com_time_series(mm_complex, mm_residue_sub, mm_ligand)
            distances_ml[substructure_name] = distance_com_time_series(ml_complex, ml_residue_sub, ml_ligand)

        # Setting common bins
        maxbin = np.ceil(max([dist.max() for dist in distances_mm.values()] + [dist.max() for dist in distances_ml.values()]))
        bins = np.linspace(0, maxbin, int(maxbin / binsize + 1))

        # Calculating Jensen-Shannon distances for substructures
        for substructure_name in residue_substructure_queries[residue.resname]:

            # Calculating density
            density_mm = np.histogram(distances_mm[substructure_name], bins=bins)[0]
            density_ml = np.histogram(distances_ml[substructure_name], bins=bins)[0]

            # Calculating Jensen-Shannon distances
            jensen_shannon_distances[f'{substructure_name} of {residue.resname.title()}{residue.resid}'] = distance.jensenshannon(density_ml, density_mm)
            
            # Calculating difference in means
            mean_diffs[f'{substructure_name} of {residue.resname.title()}{residue.resid}'] = distances_mm[substructure_name].mean() - distances_ml[substructure_name].mean()

    # Plotting Jensen-Shannon distances
    fig = plt.figure(figsize=(6,len(jensen_shannon_distances)*0.5))
    plt.barh(jensen_shannon_distances.keys(), jensen_shannon_distances.values(), color=palette['indigo'])
    plt.xlabel("Jensen-Shannon distance between MM and ML")
    plt.xlim(0, np.sqrt(np.log(2))) # Bounded by 0 and log(2) for base e

    plt.tight_layout()
    

    # Optionally saving
    if save_to:
        fig.savefig(save_to)

    # Plotting difference in means versus Jensen-Shannon distances
    if scatter:
        fig = plt.figure(figsize=(4,3))
        x = [*mean_diffs.values()]
        y = [*jensen_shannon_distances.values()]
        plt.scatter(x, y, color=palette['indigo'])
        plt.xlim(-4,4)
        plt.ylim(0,np.sqrt(np.log(2)))
        plt.vlines(0,*plt.ylim(), ls='--', color='k', alpha=0.5)
        plt.xlabel('mean(MM) - mean(ML)')
        plt.ylabel('Jensen-Shannon distance between MM and ML')
        for i, txt in enumerate(jensen_shannon_distances):
            plt.annotate(txt, (x[i], y[i]), size=6)
        plt.gca().axes.get_xaxis().set_major_formatter(tck.FormatStrFormatter('%g Å'))
        plt.tight_layout()

        if save_scatter_to:
            fig.savefig(save_scatter_to)

    return jensen_shannon_distances



# ················································································· #
# ···························· O R I E N T A T I O N S ···························· #
# ················································································· #

def plane_normal_vector(positions):
    # Define vectors in the plane of the ring
    v1 = positions[1] - positions[0]
    v2 = positions[2] - positions[0]
    
    # Normal vector is the cross product
    normal_vector = np.cross(v1, v2)
    
    # Normalize the vector
    return normal_vector / np.linalg.norm(normal_vector)

# ················································································· #

def angle_between_vectors(v1, v2):

    # Arctan 2 is bounded in [-pi,pi]
    angle_rad = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

    return np.degrees(angle_rad)

# ················································································· #

def plane_angle_time_series(universe: mda.Universe,
                            selection1: mda.AtomGroup,
                            selection2: mda.AtomGroup):

    # Looping over trajectory
    angles = []
    for timestep in tqdm(universe.trajectory):

        # Calcualing planar angles
        plane_normal_vector1 = plane_normal_vector(selection1.positions)
        plane_normal_vector2 = plane_normal_vector(selection2.positions)
        angle = angle_between_vectors(plane_normal_vector1, plane_normal_vector2)
        angles.append(angle)
    angles = np.array(angles)

    return angles

# ················································································· #

def plot_plane_angle_distributions(mm_complex: mda.Universe,
                                   ml_complex: mda.Universe,
                                   plane_query1: str,
                                   plane_query2: str,
                                   binsize: float = 5,
                                   xlims: tuple[float, float] = (2,10),
                                   size: tuple[float, float] = (3.0, 1.5),
                                   alpha: float = 0.6,
                                   save_to: str = False):
    # Making atom selections
    mm_plane1 = mm_complex.select_atoms(plane_query1)
    mm_plane2 = mm_complex.select_atoms(plane_query2)
    ml_plane1 = ml_complex.select_atoms(plane_query1)
    ml_plane2 = ml_complex.select_atoms(plane_query2)

    # Calculating angles
    mm_angles = plane_angle_time_series(mm_complex, mm_plane1, mm_plane2)
    ml_angles = plane_angle_time_series(ml_complex, ml_plane1, ml_plane2)

    # Setting bins
    bins = np.linspace(-180, 180, int(360 / binsize + 1))
    bins_mid = (bins[1:] + bins[:-1]) / 2

    # Initializing figure
    fig = plt.figure(figsize=size)
    ax = plt.gca()

    # Histograms
    plt.hist(mm_angles, bins=bins, density=True,
            alpha=alpha, label="MM", color=palette['red'])
    plt.hist(ml_angles, bins=bins, density=True,
            alpha=alpha, label="ML", color=palette['mint'])

    # Setting axis limits
    plt.xlim(-180, 180)

    # Formatting axes
    ax.axes.get_xaxis().set_major_formatter(tck.FormatStrFormatter('%g°'))
    ax.axes.get_xaxis().set_major_locator(tck.MultipleLocator(base=90))
    ax.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    
    # Optionally saving
    if save_to:
        fig.savefig(save_to)

    # Returning distribution data
    return {'mm_complex': mm_angles, 'ml_complex': ml_angles}



# ················································································· #
# ······························ S T R U C T U R E S ······························ #
# ················································································· #

def center_trajectory(universe: mda.Universe,
                      center_query):
    center = universe.select_atoms(center_query)
    not_center = universe.select_atoms(f'not ({center_query})')
    transforms = [trans.unwrap(center),
                  trans.center_in_box(center, wrap=True),
                      trans.wrap(not_center)]
    universe.trajectory.add_transformations(*transforms)


def save_representative_frame(universe: mda.Universe,
                              measure: np.array,
                              represent_as: float,
                              save_to: str):

    # Calculating difference to measure centre
    difference = np.abs(measure - represent_as)

    # Finding frame index with minimum difference
    frame = np.argmin(difference)
    universe.trajectory[frame]

    # Saving frame to file
    universe.atoms.write(save_to)
