{
  "101": {
    "task": "decrease_logp",
    "goal": "increase water solubility by decreasing LogP",
    "target_change": {
      "LogP": "decrease"
    },
    "recommended_modifications": [
      "replace hydrophobic alkyl substituent with hydroxyl, ether, amine, amide, or carbonyl-containing substituent",
      "shorten long alkyl chains",
      "remove bulky hydrophobic groups such as tert-butyl, phenyl, cycloalkyl, or long alkyl chains",
      "replace phenyl with heteroaryl such as pyridyl, pyrimidyl, or oxazolyl when chemically valid",
      "introduce one polar heteroatom into a side chain",
      "demethylate methoxy to hydroxyl if increasing polarity is acceptable",
      "replace halogenated hydrophobic substituent with a polar substituent",
      "add small polar groups such as OH, NH2, CONH2, OMe, or morpholine-like fragments"
    ],
    "avoid": [
      "adding alkyl groups",
      "adding halogens without polar compensation",
      "removing existing polar groups",
      "increasing molecular size with hydrophobic fragments"
    ]
  },
  "102": {
    "task": "increase_logp",
    "goal": "decrease water solubility by increasing LogP",
    "target_change": {
      "LogP": "increase"
    },
    "recommended_modifications": [
      "add methyl, ethyl, isopropyl, tert-butyl, cyclopropyl, cyclohexyl, phenyl, or benzyl substituent",
      "replace hydroxyl with methoxy or alkoxy",
      "mask polar donors such as OH or NH by methylation",
      "replace highly polar substituent with alkyl, aryl, halogen, trifluoromethyl, or ether-containing lipophilic group",
      "add fluorine, chlorine, bromine, CF3, or other lipophilic halogenated substituent",
      "replace heteroaryl with phenyl if scaffold similarity allows",
      "remove carboxyl, hydroxyl, amine, or sulfonamide groups when chemically valid",
      "increase aromatic or aliphatic hydrophobic surface area"
    ],
    "avoid": [
      "adding hydroxyl, carboxyl, primary amine, or sulfonamide",
      "adding multiple heteroatoms",
      "strongly increasing TPSA",
      "creating ionic groups"
    ]
  },
  "103": {
    "task": "increase_qed",
    "goal": "increase drug-likeness by improving QED",
    "target_change": {
      "QED": "increase"
    },
    "recommended_modifications": [
      "reduce excessive molecular size by trimming bulky substituents",
      "reduce excessive LogP by replacing hydrophobic fragments with moderately polar groups",
      "reduce excessive TPSA by masking or removing redundant polar groups",
      "replace reactive or uncommon groups with medicinal-chemistry-friendly bioisosteres",
      "balance HBA and HBD counts by avoiding too many donors or acceptors",
      "replace long flexible alkyl chains with compact rings or small substituents",
      "replace problematic functional groups with amide, ether, heteroaryl, or small alkyl groups",
      "adjust molecule toward moderate LogP, moderate TPSA, moderate MW, and reasonable HBA/HBD counts"
    ],
    "avoid": [
      "large increase in molecular weight",
      "extreme LogP increase or decrease",
      "adding too many HBD or HBA groups",
      "creating highly reactive, charged, or unstable motifs",
      "destroying the main scaffold"
    ]
  },
  "104": {
    "task": "decrease_qed",
    "goal": "decrease drug-likeness by lowering QED",
    "target_change": {
      "QED": "decrease"
    },
    "recommended_modifications": [
      "push LogP to an extreme by adding bulky hydrophobic fragments",
      "increase TPSA excessively by adding multiple polar groups",
      "increase HBD or HBA count beyond drug-like range",
      "add large flexible alkyl chains",
      "add bulky aromatic or polycyclic substituents",
      "introduce multiple hydroxyl, amine, carboxyl, sulfonamide, or urea groups",
      "increase molecular size and complexity while preserving validity",
      "replace compact drug-like substituents with larger less drug-like groups"
    ],
    "avoid": [
      "accidentally balancing properties toward drug-like ranges",
      "small conservative edits that improve QED",
      "invalid valence changes",
      "complete scaffold destruction unless the benchmark allows it"
    ]
  },
  "105": {
    "task": "decrease_tpsa",
    "goal": "increase permeability by decreasing TPSA",
    "target_change": {
      "TPSA": "decrease"
    },
    "recommended_modifications": [
      "remove hydroxyl, primary amine, secondary amine, carboxyl, sulfonamide, or amide groups when valid",
      "replace hydroxyl with methoxy or alkyl",
      "N-methylate amide or amine donors",
      "replace carboxylic acid with ester, amide, methyl, or bioisostere with lower TPSA",
      "replace amide with less polar linker when chemically valid",
      "mask hydrogen bond donors",
      "remove redundant heteroatoms from side chains",
      "replace polar heteroaryl with less polar aryl or alkyl substituent"
    ],
    "avoid": [
      "adding new OH, NH, COOH, CONH, SO2NH, or other high-TPSA groups",
      "increasing HBA/HBD count",
      "introducing multiple heteroatoms",
      "lowering TPSA by making the molecule invalid"
    ]
  },
  "106": {
    "task": "increase_tpsa",
    "goal": "decrease permeability by increasing TPSA",
    "target_change": {
      "TPSA": "increase"
    },
    "recommended_modifications": [
      "add hydroxyl, amine, amide, carboxyl, sulfonamide, urea, carbamate, or carbonyl-containing group",
      "demethylate methoxy to hydroxyl",
      "replace alkyl substituent with polar heteroatom-containing substituent",
      "replace phenyl with heteroaryl containing nitrogen or oxygen",
      "introduce amide or ester linker",
      "add HBA or HBD groups to side chains",
      "unmask protected polar groups",
      "replace hydrophobic substituent with morpholine, piperazine, pyridine, oxazole, or similar polar fragment"
    ],
    "avoid": [
      "masking polar groups",
      "removing heteroatoms",
      "adding only hydrophobic alkyl or halogen groups",
      "large LogP increase without TPSA gain"
    ]
  },
  "107": {
    "task": "increase_hba",
    "goal": "increase hydrogen bond acceptor count",
    "target_change": {
      "HBA": "increase"
    },
    "recommended_modifications": [
      "add ether oxygen",
      "add carbonyl group such as ketone, ester, amide, carbamate, or urea",
      "add tertiary amine when chemically valid",
      "replace phenyl with pyridyl, pyrimidyl, pyrazinyl, oxazolyl, or thiazolyl",
      "replace alkyl carbon with oxygen, nitrogen, or sulfur-containing linker",
      "add methoxy, ethoxy, morpholine, nitrile, sulfone, or sulfoxide group",
      "convert hydrocarbon side chain into heteroatom-containing side chain",
      "introduce one acceptor without excessive donor addition"
    ],
    "avoid": [
      "adding donor-only groups that do not increase HBA",
      "protonated groups that may not count as HBA",
      "removing existing acceptors",
      "adding too many donors when only HBA is needed"
    ]
  },
  "108": {
    "task": "increase_hbd",
    "goal": "increase hydrogen bond donor count",
    "target_change": {
      "HBD": "increase"
    },
    "recommended_modifications": [
      "add hydroxyl group",
      "add primary or secondary amine",
      "add amide NH",
      "add urea, sulfonamide, carbamate, or hydrazide group",
      "demethylate methoxy to hydroxyl",
      "replace tertiary amine with secondary amine if valid",
      "replace ether with alcohol if scaffold-compatible",
      "introduce NH-containing heterocycle when chemically valid"
    ],
    "avoid": [
      "N-methylating donors",
      "masking OH or NH groups",
      "adding acceptor-only groups",
      "removing existing HBD groups"
    ]
  },
  "201": {
    "task": "decrease_logp_and_increase_hba",
    "goal": "increase solubility while increasing hydrogen bond acceptors",
    "target_change": {
      "LogP": "decrease",
      "HBA": "increase"
    },
    "recommended_modifications": [
      "replace hydrophobic alkyl or aryl substituent with ether, carbonyl, tertiary amine, amide, ester, or heteroaryl group",
      "replace phenyl with pyridyl or pyrimidyl",
      "add methoxy, ether, carbonyl, morpholine, nitrile, sulfone, or tertiary amine acceptor",
      "introduce oxygen or nitrogen into a hydrophobic side chain",
      "replace alkyl chain with heteroatom-containing chain",
      "add one HBA-containing polar group while avoiding unnecessary hydrophobic expansion",
      "convert hydrocarbon substituent into amide, ester, ketone, ether, or heteroaryl substituent",
      "remove bulky hydrophobic group and replace with compact acceptor-containing group"
    ],
    "avoid": [
      "adding donor-only groups without HBA gain",
      "adding alkyl or halogen groups",
      "raising HBA using very hydrophobic fragments",
      "increasing LogP while adding acceptors"
    ]
  },
  "202": {
    "task": "increase_logp_and_increase_hba",
    "goal": "decrease solubility while increasing hydrogen bond acceptors",
    "target_change": {
      "LogP": "increase",
      "HBA": "increase"
    },
    "recommended_modifications": [
      "add lipophilic acceptor-containing group such as methoxy, ethoxy, aryl ether, ester, ketone, tertiary amide, or nitrile",
      "replace phenyl with lipophilic heteroaryl containing one acceptor",
      "add halogenated ether or CF3-substituted heteroaryl group",
      "add carbonyl-containing hydrophobic substituent",
      "introduce tertiary amine embedded in a hydrophobic fragment",
      "replace hydroxyl with alkoxy to increase LogP and preserve or add acceptor character",
      "add methoxy or ethoxy instead of hydroxyl",
      "add one HBA while compensating polarity with alkyl, aryl, halogen, or CF3"
    ],
    "avoid": [
      "adding hydroxyl, primary amine, carboxylic acid, or sulfonamide",
      "using highly polar HBA additions that lower LogP",
      "adding multiple heteroatoms without lipophilic compensation",
      "increasing HBD when not needed"
    ]
  },
  "203": {
    "task": "decrease_logp_and_increase_hbd",
    "goal": "increase solubility while increasing hydrogen bond donors",
    "target_change": {
      "LogP": "decrease",
      "HBD": "increase"
    },
    "recommended_modifications": [
      "add hydroxyl group",
      "add primary or secondary amine",
      "add amide NH, sulfonamide NH, carbamate NH, or urea NH",
      "demethylate methoxy to hydroxyl",
      "replace hydrophobic alkyl or aryl substituent with OH- or NH-containing substituent",
      "replace tertiary amine with secondary amine if chemically valid",
      "introduce polar donor-containing side chain",
      "remove hydrophobic group and introduce a compact HBD group"
    ],
    "avoid": [
      "N-methylating donors",
      "adding alkyl or halogen groups",
      "adding HBA-only groups without HBD gain",
      "masking OH or NH groups"
    ]
  },
  "204": {
    "task": "increase_logp_and_increase_hbd",
    "goal": "decrease solubility while increasing hydrogen bond donors",
    "target_change": {
      "LogP": "increase",
      "HBD": "increase"
    },
    "recommended_modifications": [
      "add one HBD group together with a hydrophobic substituent",
      "introduce anilide, secondary amide, sulfonamide, or NH-containing heterocycle in a lipophilic context",
      "add hydroxyl or NH group to an aromatic or hydrophobic substituent while also adding alkyl, aryl, halogen, or CF3",
      "replace small polar donor with bulkier lipophilic donor motif",
      "add secondary amine attached to an alkyl or aryl group",
      "add amide NH with hydrophobic N- or acyl-substitution",
      "introduce indole-like or anilide-like donor motif if scaffold-compatible",
      "increase hydrophobic surface area enough to offset the donor polarity"
    ],
    "avoid": [
      "adding multiple OH/NH groups without hydrophobic compensation",
      "adding carboxylic acid",
      "strongly increasing TPSA",
      "only adding polar donor groups that decrease LogP"
    ]
  },
  "205": {
    "task": "decrease_logp_and_decrease_tpsa",
    "goal": "increase solubility while increasing permeability",
    "target_change": {
      "LogP": "decrease",
      "TPSA": "decrease"
    },
    "recommended_modifications": [
      "remove bulky hydrophobic substituent while also masking or removing polar donor groups",
      "shorten hydrophobic alkyl chain to lower LogP without adding new polar atoms",
      "remove halogen or bulky aryl substituent to lower LogP",
      "replace high-TPSA donor group with lower-TPSA neutral group",
      "N-methylate amide or amine donor while deleting hydrophobic substituent elsewhere",
      "replace hydroxyl with methoxy only if LogP decrease is achieved elsewhere",
      "remove carboxyl or sulfonamide group and compensate LogP by deleting hydrophobic fragment",
      "replace a high-polarity group with a lower-TPSA bioisostere while reducing hydrophobic surface"
    ],
    "avoid": [
      "lowering LogP by adding hydroxyl, amine, carboxyl, or sulfonamide",
      "decreasing TPSA by adding hydrophobic groups that increase LogP",
      "adding new heteroatoms",
      "making only one-objective edits"
    ]
  },
  "206": {
    "task": "decrease_logp_and_increase_tpsa",
    "goal": "increase solubility while decreasing permeability",
    "target_change": {
      "LogP": "decrease",
      "TPSA": "increase"
    },
    "recommended_modifications": [
      "add hydroxyl, amine, amide, carboxyl, sulfonamide, urea, carbamate, or carbonyl-containing group",
      "replace hydrophobic alkyl or aryl substituent with polar heteroatom-containing substituent",
      "demethylate methoxy to hydroxyl",
      "replace phenyl with pyridyl, pyrimidyl, pyrazinyl, oxazolyl, or other heteroaryl group",
      "introduce amide, ester, ketone, ether, or sulfone linker",
      "add morpholine, piperazine, hydroxylalkyl, aminoalkyl, or amide-containing side chain",
      "remove hydrophobic substituent and replace with polar group",
      "increase HBA or HBD count while reducing hydrophobic surface area"
    ],
    "avoid": [
      "masking polar groups",
      "adding alkyl, aryl, halogen, or CF3 without polar compensation",
      "removing heteroatoms",
      "decreasing TPSA while lowering LogP"
    ]
  }
}