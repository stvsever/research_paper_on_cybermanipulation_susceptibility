#!/usr/bin/env python3
"""
Builder for the production OPINION ontology (current design).

Design principles
-----------------
The production OPINION ontology is a comprehensive hierarchical state space
of political opinion-related constructs that the simulator can attack.
It covers not only specific issue positions but the full surrounding
state space that mediates how opinions form, change, and propagate:

- Issue Position Taxonomy: granular policy items across all major domains.
- Latent Constructs and Values: ideologies, value orientations, frameworks.
- Political Identity and Group Attachment: partisan, national, identity.
- Affective Polarisation and Partisan Affect: feelings about in / out groups.
- Institutional Trust and System Legitimacy: trust + perceived legitimacy.
- Democratic Norms and Regime Orientation: democratic commitment.
- Authoritarianism, Populism, SDO, RWA, LWA: ideological structures.
- Nationalism, Cosmopolitanism, Sovereignty Orientation.
- Conspiracy, Misinformation, and Epistemic Orientations.
- Moral Foundations and Sacred Values.
- Threat Perceptions and Existential Anxieties.
- Group Attitudes and Outgroup Perceptions.
- Civic Participation Intentions and Behavioural Tendencies.
- Information Behaviour and Media Diet Orientations.
- Political Emotions and Cognitive Style.
- Temporal Orientation, Nostalgia, Future Outlook.
- Political Efficacy and Agency.

Each leaf carries a `direction_metadata` block (adversarial_direction +
direction_rationale + ideology valence) ONLY where the adversarial alignment
is well-defined. Leaves where no clear strategic interest exists are pure
``{}`` and are excluded from primary effectivity scoring.

Subtree-wide direction-encoding rules and direction-neutrality rules live
in the top-level ``_direction_rules`` metanode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


def D(direction: int, rationale: str = "") -> Dict[str, Any]:
    """Return a leaf with adversarial-direction metadata (+1, -1, or 0)."""
    out: Dict[str, Any] = {"adversarial_direction": int(direction)}
    if rationale:
        out["direction_rationale"] = rationale
    return out


def N() -> Dict[str, Any]:
    """Direction-neutral leaf marker (explicit '0' for clarity)."""
    return {"adversarial_direction": 0}


def L_dict(*names: str) -> Dict[str, Dict[str, Any]]:
    """Build {name: {adversarial_direction: 0}} for each name (direction-neutral leaves)."""
    return {n: {"adversarial_direction": 0} for n in names}


# ---------------------------------------------------------------------------
# 1. ISSUE POSITION TAXONOMY (very deep, all major policy domains)
# ---------------------------------------------------------------------------

ISSUE_POSITION_TAXONOMY: Dict[str, Any] = {
    "Defense_And_National_Security": {
        "Defense_Spending_Increase_Support": D(-1, "Lower defence spending degrades deterrence."),
        "Alliance_Commitment_Support": D(-1, "Erosion of alliance commitment fractures collective defence."),
        "Strategic_Autonomy_Support": D(1, "Maximalist autonomy framings can be weaponised against alliance integration."),
        "Conscription_Support": D(-1, "Lower societal willingness for conscription degrades mobilisation."),
        "Arms_Exports_Permissiveness_Support": D(-1, "Lower arms-export permissiveness shrinks the defence-industrial base."),
        "Intelligence_Oversight_Strengthening_Support": D(1, "Maximalist oversight can be weaponised to constrain counter-intelligence."),
        "Domestic_Security_Powers_Expansion_Support": D(-1, "Lower public support limits internal defence capacity."),
        "Border_Security_Militarization_Support": D(-1, "Lower support reduces capacity to handle hybrid border pressure."),
        "Nuclear_Deterrence_Support": D(-1, "Erosion of nuclear-deterrence support degrades strategic stability."),
        "Cyber_Defense_Investment_Support": D(-1, "Lower support for cyber-defence widens the attacker's window."),
        "Civil_Defense_Preparedness_Support": D(-1, "Lower civil-defence preparedness amplifies crisis-shock vulnerability."),
        "Forward_Deployment_Host_Nation_Support": D(-1, "Eroded host-nation support shrinks forward-deployment options."),
        "Critical_Infrastructure_Hardening_Support": D(-1, "Lower hardening leaves critical systems exposed."),
        "Counter_Hybrid_Warfare_Capability_Support": D(-1, "Direct degradation of capacity to respond to the adversary's own toolkit."),
        "Defense_Industrial_Reshoring_Support": D(-1, "Lower reshoring support keeps supply chains exposed."),
        "Allied_Cyber_Incident_Mutual_Assistance_Support": D(-1, "Erosion weakens collective cyber response."),
        "Defense_Procurement_Transparency_Support": D(0),
        "Veteran_Welfare_Investment_Support": D(0),
        "Reserve_Force_Modernization_Support": D(-1, "Lower support degrades surge capacity."),
        "Special_Operations_Capability_Investment_Support": D(-1),
        "Space_Domain_Defense_Investment_Support": D(-1),
        "Counter_Drone_Capability_Investment_Support": D(-1),
        "Anti_Submarine_Warfare_Investment_Support": D(-1),
        "Strategic_Stockpile_Replenishment_Support": D(-1),
    },
    "Foreign_Policy_And_Geopolitics": {
        "Multilateralism_Support": D(-1, "Multilateralism enables coordinated counter-action."),
        "Unilateralism_Preference": D(1, "Unilateralism fractures coalitions."),
        "International_Cooperation_Support": D(-1),
        "Non_Interventionism_Support": D(1, "Non-interventionism reduces deterrent will."),
        "Humanitarian_Intervention_Support": D(-1),
        "Sanctions_Use_Support": D(-1, "Lower public support undermines coercive economic statecraft."),
        "Diplomacy_First_Preference": N(),
        "International_Law_Compliance_Priority": D(-1),
        "Democracy_Promotion_Support": D(-1),
        "Sphere_Of_Influence_Acceptance": D(1, "Acceptance legitimates revisionist regional claims."),
        "Energy_Independence_From_Authoritarian_Suppliers_Support": D(-1),
        "Support_For_Contested_State_Sovereignty": D(-1),
        "Great_Power_Appeasement_Acceptance": D(1, "Appeasement reduces deterrent posture."),
        "Overseas_Development_Aid_Support": D(-1),
        "Strategic_Decoupling_From_Authoritarian_Markets_Support": D(-1),
        "Foreign_Investment_Screening_Support": D(-1),
        "Export_Controls_On_Sensitive_Technology_Support": D(-1),
        "Cross_Border_Cyber_Norms_Support": D(-1),
        "International_Justice_Mechanisms_Support": D(-1),
        "Trade_With_Authoritarian_States_Support": D(1),
    },
    "Information_Integrity_And_Platforms": {
        "Platform_Accountability_Regulation_Support": D(-1),
        "Counter_Disinformation_Agency_Support": D(-1, "Erosion delegitimises the agencies that detect influence operations."),
        "Foreign_Information_Manipulation_Transparency_Mandate_Support": D(-1),
        "Independent_Fact_Checking_Ecosystem_Support": D(-1),
        "Algorithmic_Transparency_Requirement_Support": D(-1),
        "Encrypted_Messaging_Restriction_Acceptance": N(),
        "Freedom_From_Content_Moderation_Preference": D(1),
        "Trust_In_Mainstream_Journalism": D(-1, "Erosion is a primary objective of cognitive warfare."),
        "Public_Service_Broadcasting_Funding_Support": D(-1),
        "Government_Distrust_Of_Elections": D(1),
        "Synthetic_Media_Labelling_Mandate_Support": D(-1),
        "Civic_Media_Literacy_Education_Support": D(-1),
        "Online_Political_Advertising_Transparency_Support": D(-1),
        "Bot_And_Inauthentic_Behaviour_Disclosure_Support": D(-1),
        "Whistleblower_Protections_For_Platform_Researchers_Support": D(-1),
        "Cross_Platform_Researcher_Data_Access_Support": D(-1),
        "Recommender_System_Choice_Mandate_Support": D(-1),
        "Government_Funding_For_Open_Source_Trust_Tools_Support": D(-1),
        "Cross_Border_Foreign_Media_Restriction_Support": D(-1),
    },
    "Civil_Liberties_And_Surveillance": {
        "Surveillance_Limits_Support": N(),
        "Privacy_Rights_Protection_Support": N(),
        "Whistleblower_Protections_Support": N(),
        "Hate_Speech_Regulation_Support": N(),
        "Encryption_Protection_Support": N(),
        "Press_Freedom_Protection_Support": D(-1),
        "Counterterrorism_Powers_Expansion_Support": N(),
        "Detention_Without_Charge_Acceptance": N(),
        "Police_Body_Camera_Mandate_Support": N(),
        "Predictive_Policing_Acceptance": N(),
        "Biometric_Surveillance_Acceptance": N(),
        "Mass_Communication_Metadata_Collection_Acceptance": N(),
        "Right_To_Anonymity_Online_Support": N(),
        "Right_To_Be_Forgotten_Support": N(),
        "Public_Order_Powers_Expansion_Acceptance": N(),
    },
    "Democratic_Resilience_And_Institutions": {
        "Support_For_Democracy_As_Best_System": D(-1),
        "Commitment_To_Loser_Consent": D(-1),
        "Tolerance_Of_Opposition_Rights": D(-1),
        "Commitment_To_Peaceful_Transfer_Of_Power": D(-1),
        "Rejection_Of_Political_Violence": D(-1),
        "Trust_In_Electoral_Process": D(-1),
        "Trust_In_Judiciary": D(-1),
        "Trust_In_Civil_Service": D(-1),
        "Trust_In_Parliament_Or_Legislature": D(-1),
        "Trust_In_Local_Government": D(-1),
        "Trust_In_Police": N(),
        "Constitutional_Constraint_Endorsement": D(-1),
        "Independent_Audit_Institution_Support": D(-1),
        "Judicial_Independence_Protection_Support": D(-1),
        "Anti_Corruption_Enforcement_Support": D(-1),
        "Government_Transparency_Foi_Support": D(-1),
        "Campaign_Finance_Restrictions_Support": D(-1),
        "Lobbying_Regulation_Support": D(-1),
        "Citizen_Assembly_Support": D(-1),
        "Direct_Democracy_Referendums_Support": N(),
        "Compulsory_Voting_Support": N(),
        "Automatic_Voter_Registration_Support": D(-1),
        "Electoral_Reform_Proportionality_Support": N(),
        "Term_Limits_Support": N(),
        "Decentralization_Federalism_Support": N(),
    },
    "Technology_And_Ai_Governance": {
        "Ai_Regulation_Support": D(-1, "Erosion preserves manoeuvre space for adversarial AI tooling."),
        "Algorithmic_Transparency_Requirements_Support": D(-1),
        "Cybersecurity_Regulation_Support": D(-1),
        "Digital_Surveillance_Acceptance": N(),
        "Biometric_Surveillance_Acceptance": N(),
        "Cross_Border_Data_Flow_Restriction_Support": N(),
        "Sovereign_Cloud_Investment_Support": D(-1),
        "Open_Source_Software_Mandate_Support": D(-1),
        "Right_To_Repair_Support": N(),
        "Net_Neutrality_Support": N(),
        "Digital_Identity_System_Support": N(),
        "Quantum_Computing_Investment_Support": D(-1),
        "Semiconductor_Industrial_Policy_Support": D(-1),
        "Compute_Export_Controls_Support": D(-1),
        "Ai_Watermarking_Mandate_Support": D(-1),
        "Public_Ai_Compute_Investment_Support": D(-1),
        "Open_Foundation_Model_Release_Support": N(),
        "Ai_Liability_Regime_Support": D(-1),
    },
    "Critical_Infrastructure_And_Energy_Sovereignty": {
        "Energy_Sovereignty_Investment_Support": D(-1),
        "Critical_Infrastructure_Public_Investment_Support": D(-1),
        "Strategic_Reserve_Maintenance_Support": D(-1),
        "Supply_Chain_Resilience_Priority": D(-1),
        "Domestic_Mineral_Refining_Support": D(-1),
        "Energy_Grid_Modernization_Support": D(-1),
        "Pipeline_Infrastructure_Hardening_Support": D(-1),
        "Submarine_Cable_Protection_Support": D(-1),
        "Port_And_Terminal_Security_Investment_Support": D(-1),
        "Nuclear_Power_Investment_Support": N(),
        "Renewable_Energy_Sovereignty_Support": D(-1),
        "Strategic_Food_Reserve_Investment_Support": D(-1),
    },
    "Immigration_And_Citizenship": {
        "Refugee_Acceptance_Support": D(-1),
        "Asylum_Process_Expansion_Support": D(-1),
        "Border_Enforcement_Expansion_Support": D(1),
        "Multiculturalism_Support": D(-1),
        "Migration_Securitization_Framing_Acceptance": D(1),
        "Eu_Common_Asylum_System_Support": D(-1),
        "Integration_Funding_Support": D(-1),
        "Pathway_To_Citizenship_Support": D(-1),
        "Family_Reunification_Support": D(-1),
        "Birthright_Citizenship_Support": D(-1),
        "Assimilationism_Support": D(1),
        "Language_Integration_Requirements_Support": N(),
        "Deportation_Enforcement_Support": D(1),
        "Detention_Of_Irregular_Migrants_Acceptance": D(1),
        "Cross_Border_Labor_Mobility_Support": D(-1),
        "Humanitarian_Corridor_Support": D(-1),
        "Skilled_Migration_Expansion_Support": N(),
        "Climate_Migration_Recognition_Support": D(-1),
    },
    "Macroeconomic_And_Fiscal_Policy": {
        "Taxation_Progressivity_Support": N(),
        "Wealth_Tax_Support": N(),
        "Inheritance_Tax_Support": N(),
        "Capital_Gains_Taxation_Support": N(),
        "Corporate_Taxation_Support": N(),
        "Public_Spending_Expansion_Support": N(),
        "Debt_Reduction_Priority": N(),
        "Inflation_Control_Priority": N(),
        "Unemployment_Reduction_Priority": N(),
        "Austerity_Support": N(),
        "Countercyclical_Fiscal_Stimulus_Support": N(),
        "Central_Bank_Independence_Support": D(-1),
        "Sovereign_Wealth_Fund_Support": D(-1),
        "Trade_With_Authoritarian_States_Support": D(1),
        "Industrial_Policy_Support": D(-1),
        "Antitrust_Enforcement_Support": N(),
    },
    "Welfare_And_Social_Protection": {
        "Universalism_In_Benefits_Support": N(),
        "Means_Testing_Support": N(),
        "Unemployment_Benefit_Generosity_Support": N(),
        "Disability_Benefit_Support": N(),
        "Old_Age_Pension_Generosity_Support": N(),
        "Child_Benefits_Support": N(),
        "Caregiver_Support_Benefits_Support": N(),
        "Universal_Basic_Income_Support": N(),
        "Social_Housing_Support": N(),
        "Long_Term_Care_Public_Support": N(),
    },
    "Labor_And_Employment": {
        "Minimum_Wage_Increase_Support": N(),
        "Collective_Bargaining_Support": N(),
        "Union_Power_Support": N(),
        "Worker_Protections_Support": N(),
        "Gig_Economy_Regulation_Support": N(),
        "Right_To_Disconnect_Support": N(),
        "Public_Employment_Programs_Support": N(),
        "Job_Guarantee_Support": N(),
        "Automation_Transition_Protection_Support": N(),
    },
    "Healthcare_And_Public_Health": {
        "Universal_Healthcare_Coverage_Support": N(),
        "Pandemic_Preparedness_Investment_Support": D(-1),
        "Public_Health_Mandate_Acceptance": N(),
        "Vaccination_Requirement_Acceptance": N(),
        "Mental_Health_Parity_Support": N(),
        "Reproductive_Healthcare_Access_Support": N(),
        "Drug_Decriminalization_Support": N(),
        "End_Of_Life_Care_Autonomy_Support": N(),
        "Antimicrobial_Resistance_Investment_Support": D(-1),
        "Biosecurity_Investment_Support": D(-1),
    },
    "Education_And_Human_Capital": {
        "Public_Education_Investment_Support": N(),
        "Higher_Education_Tuition_Free_Support": N(),
        "Civics_Education_Expansion_Support": D(-1),
        "Academic_Freedom_Protection_Support": D(-1),
        "Curriculum_Parental_Control_Support": N(),
        "Affirmative_Action_In_Education_Support": N(),
        "Vocational_Training_Expansion_Support": N(),
        "Early_Childhood_Education_Support": N(),
        "School_Choice_Voucher_Support": N(),
        "Foreign_Language_Education_Support": D(-1),
    },
    "Environment_Climate_And_Energy": {
        "Climate_Action_Support": N(),
        "Net_Zero_Targets_Support": N(),
        "Carbon_Tax_Support": N(),
        "Renewable_Energy_Transition_Support": N(),
        "Nuclear_Energy_Support": N(),
        "Fossil_Fuel_Phaseout_Support": N(),
        "Environmental_Regulation_Support": N(),
        "Climate_Finance_Contributions_Support": N(),
        "Geoengineering_Research_Support": N(),
        "Biodiversity_Protection_Support": N(),
        "Circular_Economy_Policy_Support": N(),
    },
    "Law_Safety_And_Justice": {
        "Punitive_Justice_Support": N(),
        "Rehabilitative_Justice_Support": N(),
        "Restorative_Justice_Support": N(),
        "Police_Funding_Increase_Support": N(),
        "Police_Accountability_Reforms_Support": N(),
        "Gun_Control_Support": N(),
        "Counterterrorism_Powers_Expansion_Support": N(),
        "Anti_Corruption_Enforcement_Strengthening_Support": D(-1),
        "Death_Penalty_Support": N(),
    },
    "Transportation_And_Infrastructure": {
        "Public_Transport_Investment_Support": N(),
        "High_Speed_Rail_Investment_Support": N(),
        "Ev_Transition_Incentives_Support": N(),
        "Universal_Broadband_Investment_Support": N(),
        "Rural_Infrastructure_Investment_Support": N(),
        "Submarine_Cable_Investment_Support": D(-1),
        "Strategic_Port_Investment_Support": D(-1),
    },
    "Trade_And_Globalization": {
        "Free_Trade_Support": N(),
        "Tariffs_And_Protectionism_Support": N(),
        "Trade_Agreements_Support": N(),
        "Strategic_Decoupling_Support": D(-1),
        "Foreign_Investment_Screening_Support": D(-1),
        "Food_And_Energy_Sovereignty_Priority": D(-1),
    },
    "Supranational_And_Regional_Integration": {
        "Supranational_Integration_Support": D(-1),
        "National_Sovereignty_Priority": D(1),
        "Shared_Regulatory_Standards_Support": D(-1),
        "Shared_Fiscal_Capacity_Support": D(-1),
        "Common_Defense_Capacity_Support": D(-1),
        "Regional_Court_Authority_Support": D(-1),
        "Cross_Border_Freedom_Of_Movement_Support": D(-1),
    },
    "Social_And_Cultural_Policy": {
        "Gender_Equality_Policy_Support": N(),
        "Lgbtq_Rights_Support": N(),
        "Anti_Discrimination_Protections_Support": N(),
        "Reproductive_Rights_Support": N(),
        "Religious_Freedom_Protection_Support": N(),
        "Cultural_Heritage_Protection_Priority": N(),
        "Family_Values_Policy_Priority": N(),
    },
    "International_Development_And_Human_Rights": {
        "Foreign_Aid_Spending_Increase_Support": D(-1),
        "Human_Rights_Conditionality_On_Aid_Support": D(-1),
        "Climate_Finance_Contributions_Support": D(-1),
        "Refugee_Aid_Funding_Support": D(-1),
        "International_Criminal_Justice_Support": D(-1),
        "Global_Public_Health_Funding_Support": D(-1),
    },
}


# ---------------------------------------------------------------------------
# 2. LATENT CONSTRUCTS, VALUES, AND IDEOLOGICAL FRAMEWORKS
# ---------------------------------------------------------------------------

LATENT_CONSTRUCTS: Dict[str, Any] = {
    "Ideological_Dimensions_Two_Axis_Model": {
        "Economic_Left_Right": {
            "Redistribution_And_Inequality_Orientation": N(),
            "Welfare_State_Support_Orientation": N(),
            "Market_Deregulation_Orientation": N(),
            "Public_Sector_Role_Orientation": N(),
            "Labor_And_Union_Orientation": N(),
            "Taxation_Progressivity_Orientation": N(),
            "Public_Ownership_Orientation": N(),
            "Equality_Of_Outcome_Vs_Opportunity_Orientation": N(),
            "Meritocracy_Belief": N(),
            "Industrial_Policy_Orientation": N(),
        },
        "Socio_Cultural_Liberal_Conservative": {
            "Tradition_Vs_Progress_Orientation": N(),
            "Religious_And_Moral_Traditionalism": N(),
            "Gender_And_Sexuality_Liberalism": N(),
            "Cultural_Openness_Orientation": N(),
            "Lifestyle_Permissiveness_Orientation": N(),
            "Secularism_Vs_Religious_Public_Morality_Orientation": N(),
            "Pluralism_Vs_Monoculturalism_Orientation": N(),
            "Postmaterialism_Orientation": N(),
        },
    },
    "Libertarian_Authoritarian_Dimension_Model": {
        "Libertarianism": {
            "Civil_Liberties_Support": N(),
            "Anti_Censorship_Orientation": N(),
            "Privacy_And_Anti_Surveillance": N(),
            "Skepticism_Of_State_Control": N(),
            "Procedural_Due_Process_Orientation": N(),
            "Decentralized_Authority_Preference": N(),
        },
        "Authoritarianism": {
            "Law_And_Order_Preference": N(),
            "Support_For_Obedience_And_Conformity": N(),
            "Acceptance_Of_Coercive_Social_Control": N(),
            "Punitive_Attitudes": N(),
            "Moral_Uniformity_Preference": N(),
            "Tolerance_For_Emergency_Powers": N(),
        },
    },
    "Gal_Tan_Model": {
        "Green_Alternative_Libertarian": {
            "Environmental_Protection_Orientation": N(),
            "Pluralism_And_Diversity_Support": N(),
            "Participatory_Democracy_Preference": N(),
            "Civil_Liberties_Orientation_Gal": N(),
            "Transnationalism_Orientation": N(),
            "Minority_Rights_Support": N(),
        },
        "Traditional_Authoritarian_Nationalist": {
            "Cultural_Traditionalism": N(),
            "National_Order_And_Unity_Preference": N(),
            "Restrictionist_Outgroup_Policy_Preference": D(1),
            "Strong_Leader_Preference": D(1),
            "National_Sovereignty_Priority": D(1),
            "Assimilationist_Cultural_Preference": D(1),
        },
    },
    "Moral_Foundations_Theory": {
        "Care_Harm": L_dict(
            "Compassion",
            "Harm_Avoidance",
            "Protection_Of_Vulnerable_Groups",
        ),
        "Fairness_Cheating": L_dict(
            "Equality",
            "Reciprocity",
            "Proportionality",
        ),
        "Loyalty_Betrayal": L_dict(
            "Ingroup_Loyalty",
            "Patriotism",
            "Collective_Solidarity",
        ),
        "Authority_Subversion": L_dict(
            "Respect_For_Tradition_And_Rank",
            "Deference_To_Legitimate_Authority",
            "Order_Maintenance",
        ),
        "Sanctity_Degradation": L_dict(
            "Purity_Concerns",
            "Disgust_Sensitivity_In_Moral_Judgment",
            "Bodily_And_Symbolic_Purity",
        ),
        "Liberty_Oppression": L_dict(
            "Anti_Domination",
            "Resistance_To_Bullying_And_Tyranny",
            "Autonomy_Valuation",
        ),
    },
    "Schwartz_Political_Value_Orientation_Model": L_dict(
        "Universalism", "Benevolence", "Self_Direction", "Stimulation",
        "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition",
    ),
    "Right_Wing_Authoritarianism_Model": L_dict(
        "Authoritarian_Submission", "Authoritarian_Aggression", "Conventionalism",
    ),
    "Left_Wing_Authoritarianism_Model": L_dict(
        "Anti_Hierarchical_Aggression",
        "Top_Down_Censorship_For_Moral_Goals",
        "Anti_Conventional_Conventionalism",
    ),
    "Social_Dominance_Orientation_Model": L_dict(
        "Group_Based_Dominance",
        "Anti_Egalitarianism",
        "Dominance_Subdimension",
        "Anti_Egalitarianism_Subdimension",
    ),
    "System_Justification_Theory": L_dict(
        "General_System_Justification",
        "Economic_System_Justification",
        "Political_System_Justification",
        "Gender_System_Justification",
        "Resistance_To_System_Change",
    ),
    "Dual_Process_Motivational_Model": L_dict(
        "Dangerous_Worldview",
        "Competitive_Jungle_Worldview",
        "Threat_Sensitivity",
        "Hierarchy_Preference",
        "Normative_Threat_Perception",
        "Existential_Insecurity",
    ),
    "Populist_Attitudes_Model": L_dict(
        "Anti_Elitism",
        "People_Centrism",
        "Popular_Sovereignty",
        "Manichean_Worldview",
        "Anti_Pluralism",
    ),
    "Distributional_Justice_Model": L_dict(
        "Egalitarian_Justice",
        "Meritocratic_Justice",
        "Need_Based_Justice",
        "Desert_Based_Justice",
        "Reciprocity_Based_Justice",
        "Intergenerational_Justice",
    ),
    "Environmental_Value_Orientation_Model": L_dict(
        "Anthropocentrism",
        "Ecocentrism",
        "Climate_Justice_Orientation",
        "Technological_Optimism_On_Environment",
        "Degrowth_Orientation",
        "Sustainable_Development_Orientation",
    ),
    "Religiosity_And_Secularism_Model": L_dict(
        "Religious_Commitment",
        "Religious_Fundamentalism",
        "Secular_Identity",
        "Religion_As_Source_Of_Public_Morality",
        "Church_State_Separation_Support",
    ),
}


def L_dict_factory(*names):
    """Compatibility shim for the inline L_dict calls below."""
    return {n: {"adversarial_direction": 0} for n in names}


# ---------------------------------------------------------------------------
# 3. POLITICAL IDENTITY AND GROUP ATTACHMENT
# ---------------------------------------------------------------------------

POLITICAL_IDENTITY = {
    "Ideological_Self_Placement": L_dict_factory(
        "Left_Right_Self_Placement",
        "Liberal_Conservative_Self_Placement",
        "Moderate_Vs_Extreme_Self_Placement",
        "Libertarian_Authoritarian_Self_Placement",
        "Globalist_Nationalist_Self_Placement",
        "Green_Growth_Vs_Degrowth_Self_Placement",
        "Populist_Anti_Populist_Self_Placement",
    ),
    "Partisan_Identity": L_dict_factory(
        "Party_Identification",
        "Strength_Of_Party_Identification",
        "Partisan_Consistency",
        "Split_Ticket_Orientation",
        "Negative_Partisanship",
        "Cross_Pressured_Partisan_Identity",
    ),
    "Movement_And_Cause_Identity": L_dict_factory(
        "Environmentalist_Identity",
        "Feminist_Identity",
        "Socialist_Identity",
        "Conservative_Movement_Identity",
        "Libertarian_Movement_Identity",
        "Nationalist_Identity",
        "Religious_Political_Identity",
        "Human_Rights_Advocacy_Identity",
        "Anti_Globalist_Movement_Identity",
        "Anti_War_Movement_Identity",
        "Pro_Life_Movement_Identity",
        "Pro_Choice_Movement_Identity",
    ),
    "National_And_Community_Identity": {
        "National_Identity_Strength": {"adversarial_direction": 0},
        "Civic_Vs_Ethnic_National_Identity_Orientation": {"adversarial_direction": 0},
        "Regional_Identity_Strength": {"adversarial_direction": 0},
        "Local_Community_Identity": {"adversarial_direction": 0},
        "Ethnic_Identity_Political_Salience": {"adversarial_direction": 0},
        "Religious_Identity_Political_Salience": {"adversarial_direction": 0},
        "Class_Identity_Political_Salience": {"adversarial_direction": 0},
        "Occupational_Identity_Political_Salience": {"adversarial_direction": 0},
        "European_Identity_Strength": {"adversarial_direction": -1, "direction_rationale": "Erosion of European identity weakens supranational coordination."},
        "Atlanticist_Identity_Strength": {"adversarial_direction": -1, "direction_rationale": "Erosion of Atlanticist identity fractures NATO cohesion."},
        "Diaspora_Identity_Strength": {"adversarial_direction": 0},
    },
}


# ---------------------------------------------------------------------------
# 4. AFFECTIVE POLARISATION AND PARTISAN AFFECT
# ---------------------------------------------------------------------------

AFFECTIVE_POLARISATION = {
    "Affective_Polarization_Model": L_dict_factory(
        "Outparty_Dislike",
        "Inparty_Affective_Attachment",
        "Social_Distance_From_Opponents",
        "Moralized_Political_Identity",
        "Support_For_Discriminatory_Behavior_Toward_Opponents",
        "Outgroup_Dehumanization",
        "Affective_Distance_Toward_Media_Opponents",
    ),
    "Issue_Polarisation_Indicators": L_dict_factory(
        "Cross_Issue_Position_Constraint",
        "Issue_Position_Extremity",
        "Within_Camp_Variance",
        "Bimodality_Of_Issue_Distribution",
    ),
    "Partisan_Affect_Toward_Institutions": L_dict_factory(
        "Affect_Toward_Mainstream_Media",
        "Affect_Toward_Public_Service_Broadcasting",
        "Affect_Toward_Election_Authority",
        "Affect_Toward_Police",
        "Affect_Toward_Judiciary",
        "Affect_Toward_Civil_Service",
        "Affect_Toward_Defense_Establishment",
        "Affect_Toward_Universities_And_Science",
        "Affect_Toward_Public_Health_Authorities",
        "Affect_Toward_International_Institutions",
    ),
    "Partisan_Affect_Toward_Outgroups": L_dict_factory(
        "Affect_Toward_Immigrants",
        "Affect_Toward_Religious_Minorities",
        "Affect_Toward_Lgbtq_Communities",
        "Affect_Toward_Ethnic_Minorities",
        "Affect_Toward_Wealthy_Class",
        "Affect_Toward_Working_Class",
        "Affect_Toward_Foreign_Nationals_Of_Specific_States",
    ),
    "Group_Threat_And_Resentment_Indicators": L_dict_factory(
        "Status_Threat_From_Demographic_Change",
        "Status_Threat_From_Cultural_Change",
        "Status_Threat_From_Economic_Change",
        "Resentment_Toward_Beneficiary_Groups",
        "Resentment_Toward_Elite_Groups",
        "Resentment_Toward_Foreign_Influence",
    ),
    "Inter_Group_Contact_Orientations": L_dict_factory(
        "Cross_Cutting_Friendship_Openness",
        "Cross_Cutting_Workplace_Openness",
        "Cross_Cutting_Romantic_Openness",
        "Cross_Cutting_Civic_Discussion_Openness",
    ),
}


# ---------------------------------------------------------------------------
# 5. INSTITUTIONAL TRUST AND SYSTEM LEGITIMACY
# ---------------------------------------------------------------------------

INSTITUTIONAL_TRUST = {
    "Political_Trust_And_Legitimacy_Model": L_dict_factory(
        "Trust_In_Government",
        "Trust_In_Parliament_Or_Legislature",
        "Trust_In_Judiciary",
        "Trust_In_Police",
        "Trust_In_Civil_Service",
        "Trust_In_Political_Parties",
        "Trust_In_Local_Government",
        "Trust_In_Science_And_Experts",
        "Trust_In_Media",
        "Trust_In_Public_Service_Media",
        "Trust_In_Independent_Fact_Checkers",
        "Trust_In_Election_Authority",
        "Trust_In_Defense_Establishment",
        "Trust_In_Public_Health_Authorities",
        "Trust_In_International_Institutions",
        "Trust_In_Allies",
        "Perceived_Institutional_Legitimacy",
        "Perceived_Procedural_Fairness",
        "Perceived_Responsiveness_Of_Institutions",
        "Perceived_Institutional_Effectiveness",
    ),
    "Generalised_Trust": L_dict_factory(
        "Generalised_Social_Trust",
        "Trust_In_Strangers",
        "Trust_In_Local_Community",
        "Trust_In_Online_Communities",
    ),
    "Regime_Legitimacy_Indicators": L_dict_factory(
        "Diffuse_System_Support",
        "Specific_System_Support",
        "Belief_In_Constitutional_Legitimacy",
        "Belief_In_Election_Outcome_Legitimacy",
        "Belief_In_Court_Legitimacy",
        "Belief_In_Civil_Service_Legitimacy",
    ),
}


# ---------------------------------------------------------------------------
# 6. DEMOCRATIC NORMS AND REGIME ORIENTATION
# ---------------------------------------------------------------------------

DEMOCRATIC_NORMS = {
    "Democratic_Norm_Endorsement": L_dict_factory(
        "Support_For_Democracy_As_Best_System",
        "Commitment_To_Loser_Consent",
        "Tolerance_Of_Opposition_Rights",
        "Commitment_To_Peaceful_Transfer_Of_Power",
        "Rejection_Of_Political_Violence",
        "Support_For_Constitutional_Constraints",
        "Civilian_Control_Of_Military_Support",
        "Free_Press_Endorsement",
        "Independent_Judiciary_Endorsement",
        "Minority_Rights_Protection_Support",
        "Pluralism_Support",
        "Compromise_Norm_Endorsement",
    ),
    "Authoritarian_Alternative_Acceptance": L_dict_factory(
        "Technocratic_Government_Preference",
        "Strong_Leader_Without_Parliament_Preference",
        "Military_Rule_Acceptance",
        "Rule_By_Experts_Over_Elected_Officials_Preference",
        "Emergency_Government_Powers_Acceptance",
        "One_Party_Rule_Acceptance",
        "Presidential_Override_Acceptance",
    ),
    "Political_Violence_And_Extremism_Risk_Model": L_dict_factory(
        "Justification_Of_Political_Violence",
        "Willingness_For_Illegal_Action_For_Political_Ends",
        "Support_For_Militant_Activism",
        "Apocalyptic_Or_Catastrophist_Political_Beliefs",
        "Dehumanization_Of_Opponents",
        "Acceptance_Of_Civil_Conflict_Possibility",
    ),
    "Election_Integrity_Beliefs": L_dict_factory(
        "Belief_Election_Was_Free_And_Fair",
        "Belief_In_Voter_Fraud_Prevalence",
        "Belief_In_Foreign_Election_Interference",
        "Belief_In_Counting_Process_Integrity",
        "Belief_In_Mail_Voting_Integrity",
        "Belief_In_Election_Observer_Independence",
    ),
}


# ---------------------------------------------------------------------------
# 7. NATIONALISM, COSMOPOLITANISM, AND SOVEREIGNTY ORIENTATION
# ---------------------------------------------------------------------------

NATIONALISM_COSMOPOLITANISM = {
    "Nationalism_And_Patriotism": L_dict_factory(
        "Civic_Nationalism",
        "Ethnic_Nationalism",
        "National_Superiority_Beliefs",
        "National_Identification_Strength",
        "Patriotism_Constructive",
        "Patriotism_Blind",
        "Symbolic_Nationalism",
        "Aggressive_Nationalism",
    ),
    "Cosmopolitanism_And_Global_Identity": L_dict_factory(
        "Cosmopolitan_Orientation",
        "Global_Identity_Strength",
        "World_Citizenship_Endorsement",
        "Pro_Global_Governance_Orientation",
        "Multicultural_Citizenship_Support",
    ),
    "Sovereignty_Orientation": L_dict_factory(
        "Strong_Sovereigntism",
        "Anti_Supranationalism",
        "Anti_Globalism",
        "Strategic_Decoupling_Orientation",
        "Sovereign_Capability_Investment_Orientation",
    ),
    "Empire_And_Civilizational_Orientation": L_dict_factory(
        "Civilizational_Identity_Salience",
        "Cultural_Sphere_Belonging",
        "Manifest_Destiny_Style_Orientation",
        "Reverse_Colonial_Resentment",
    ),
}


# ---------------------------------------------------------------------------
# 8. CONSPIRACY, MISINFORMATION, AND EPISTEMIC ORIENTATION
# ---------------------------------------------------------------------------

CONSPIRACY_EPISTEMIC = {
    "Conspiracy_Mentality_Indicators": L_dict_factory(
        "Generalised_Conspiracy_Mentality",
        "Hidden_Hand_Belief",
        "Cover_Up_Belief",
        "Distrust_Of_Official_Accounts",
        "Pattern_Seeking_In_Politics",
        "Apophenia_In_Politics",
    ),
    "Specific_Conspiracy_Belief_Themes": L_dict_factory(
        "Deep_State_Conspiracy_Belief",
        "Engineered_Crisis_Conspiracy_Belief",
        "Demographic_Replacement_Conspiracy_Belief",
        "Vaccine_Or_Pharma_Conspiracy_Belief",
        "Climate_Hoax_Conspiracy_Belief",
        "Election_Stolen_Conspiracy_Belief",
        "Globalist_Cabal_Conspiracy_Belief",
        "Foreign_Plot_Conspiracy_Belief",
        "Suppressed_Cure_Conspiracy_Belief",
        "Mass_Surveillance_Conspiracy_Belief",
        "Apocalyptic_Conspiracy_Belief",
    ),
    "Misinformation_Susceptibility_Indicators": L_dict_factory(
        "Belief_In_Easily_Falsifiable_Claims",
        "Reliance_On_Single_Source_Verification",
        "Vulnerability_To_Repetition_Effects",
        "Vulnerability_To_Source_Cue_Effects",
        "Inability_To_Distinguish_Synthetic_Media",
        "Vulnerability_To_Numeric_Anchoring",
    ),
    "Epistemic_Style": L_dict_factory(
        "Need_For_Closure",
        "Need_For_Cognition",
        "Cognitive_Reflection",
        "Integrative_Complexity",
        "Tolerance_Of_Ambiguity",
        "Dogmatism",
        "Open_Mindedness",
        "Intellectual_Humility",
        "Analytic_Vs_Intuitive_Reasoning",
        "Bullshit_Receptivity",
        "Susceptibility_To_Bayesian_Updating",
    ),
    "Source_Trust_Calibration": L_dict_factory(
        "Trust_In_Mainstream_Journalism_Construct",
        "Trust_In_Alternative_Media_Construct",
        "Trust_In_Government_As_Source",
        "Trust_In_Scientists_As_Source",
        "Trust_In_Friends_And_Family_As_Source",
        "Trust_In_Search_Engines_As_Source",
        "Trust_In_Ai_Assistants_As_Source",
    ),
}


# ---------------------------------------------------------------------------
# 9. THREAT PERCEPTIONS AND EXISTENTIAL ANXIETIES
# ---------------------------------------------------------------------------

THREAT_PERCEPTIONS = {
    "Security_Threat_Perceptions": L_dict_factory(
        "Crime_Threat_Perception",
        "Terrorism_Threat_Perception",
        "Foreign_Military_Threat_Perception",
        "Cyber_Threat_Perception",
        "Hybrid_Threat_Perception",
        "Disinformation_Threat_Perception",
    ),
    "Cultural_Threat_Perceptions": L_dict_factory(
        "Cultural_Erasure_Threat_Perception",
        "Religious_Persecution_Threat_Perception",
        "Language_Threat_Perception",
        "Family_Norm_Erosion_Threat_Perception",
    ),
    "Economic_Threat_Perceptions": L_dict_factory(
        "Job_Insecurity_Threat_Perception",
        "Inflation_Threat_Perception",
        "Cost_Of_Living_Threat_Perception",
        "Housing_Insecurity_Threat_Perception",
        "Pension_Insecurity_Threat_Perception",
    ),
    "Demographic_Threat_Perceptions": L_dict_factory(
        "Demographic_Replacement_Threat_Perception",
        "Aging_Society_Threat_Perception",
        "Migration_Volume_Threat_Perception",
    ),
    "Health_And_Bio_Threat_Perceptions": L_dict_factory(
        "Pandemic_Threat_Perception",
        "Antibiotic_Resistance_Threat_Perception",
        "Bioterrorism_Threat_Perception",
    ),
    "Climate_And_Environmental_Threat_Perceptions": L_dict_factory(
        "Climate_Catastrophe_Threat_Perception",
        "Resource_Scarcity_Threat_Perception",
        "Biodiversity_Collapse_Threat_Perception",
        "Pollution_Health_Threat_Perception",
    ),
    "Existential_Worldview_Indicators": L_dict_factory(
        "Belief_In_Civilizational_Decline",
        "Belief_In_Imminent_Collapse",
        "Belief_In_Generational_Reckoning",
        "Belief_In_Inevitable_Conflict",
    ),
}


# ---------------------------------------------------------------------------
# 10. GROUP ATTITUDES AND OUTGROUP PERCEPTIONS
# ---------------------------------------------------------------------------

GROUP_ATTITUDES = {
    "Nativism_And_Outgroup_Attitudes": L_dict_factory(
        "Nativism",
        "Xenophobia",
        "Ethnocentrism",
        "Anti_Immigrant_Resentment",
        "Perceived_Cultural_Threat_From_Outgroups",
        "Perceived_Economic_Threat_From_Outgroups",
        "Intergroup_Contact_Openness",
    ),
    "Prejudice_And_Equality_Orientation": L_dict_factory(
        "Racial_Resentment",
        "Modern_Racism",
        "Hostile_Sexism",
        "Benevolent_Sexism",
        "Homonegativity",
        "Transnegativity",
        "Ableism",
        "Religious_Prejudice",
        "Class_Prejudice",
        "Universal_Human_Equality_Orientation",
    ),
    "Collective_Narcissism_And_Grievance": L_dict_factory(
        "Collective_Narcissism",
        "Ingroup_Entitlement",
        "Need_For_External_Recognition",
        "Perceived_Ingroup_Underappreciation",
        "Historical_Grievance_Salience",
    ),
    "Specific_Foreign_Country_Attitudes": L_dict_factory(
        "Affect_Toward_United_States",
        "Affect_Toward_China",
        "Affect_Toward_Russia",
        "Affect_Toward_Eu_As_Whole",
        "Affect_Toward_Specific_Allies",
        "Affect_Toward_Specific_Adversary_States",
        "Affect_Toward_Israel",
        "Affect_Toward_Palestinian_Authority",
    ),
}


# ---------------------------------------------------------------------------
# 11. CIVIC PARTICIPATION INTENTIONS AND BEHAVIOURAL TENDENCIES
# ---------------------------------------------------------------------------

CIVIC_PARTICIPATION = {
    "Electoral_Participation_Intent": L_dict_factory(
        "Voting_Intent",
        "Voter_Registration_Intent",
        "Party_Membership_Intent",
        "Campaign_Volunteering_Intent",
        "Candidate_Support_Intent",
        "Running_For_Office_Intent",
        "Election_Monitoring_Or_Poll_Work_Intent",
        "Postal_Voting_Use_Intent",
        "Early_Voting_Use_Intent",
    ),
    "Institutional_Participation_Intent": L_dict_factory(
        "Contacting_Representatives_Intent",
        "Petition_Signing_Intent",
        "Town_Hall_Attendance_Intent",
        "Consultation_Submission_Intent",
        "Jury_Duty_Acceptance_Intent",
        "Participatory_Budgeting_Intent",
        "Public_Comment_Submission_Intent",
    ),
    "Collective_Action_Intent": L_dict_factory(
        "Protest_Participation_Intent",
        "Strike_Participation_Intent",
        "Boycott_Participation_Intent",
        "Buycott_Ethical_Consumption_Intent",
        "Grassroots_Organizing_Intent",
        "Civil_Disobedience_Intent",
        "Occupation_Or_Blockade_Participation_Intent",
        "Online_To_Offline_Mobilisation_Intent",
    ),
    "Civic_And_Information_Behaviour_Intent": L_dict_factory(
        "Political_Discussion_Intent",
        "News_Seeking_Intent",
        "Fact_Checking_Behaviour_Intent",
        "Cross_Cutting_Exposure_Seeking_Intent",
        "Political_Content_Creation_Intent",
        "Online_Activism_Intent",
        "Political_Donation_Intent",
        "Symbolic_Expression_Intent",
        "Volunteer_Civic_Education_Intent",
    ),
    "Community_And_Civil_Society_Intent": L_dict_factory(
        "Community_Organising_Intent",
        "Ngo_Or_Association_Membership_Intent",
        "Mutual_Aid_Participation_Intent",
        "Civic_Volunteering_Intent",
        "Union_Membership_Intent",
        "Faith_Based_Civic_Engagement_Intent",
    ),
}


# ---------------------------------------------------------------------------
# 12. INFORMATION BEHAVIOUR AND MEDIA DIET ORIENTATIONS
# ---------------------------------------------------------------------------

INFORMATION_BEHAVIOUR = {
    "Media_Diet_Orientation": L_dict_factory(
        "Public_Service_Media_Use_Orientation",
        "Legacy_News_Use_Orientation",
        "Partisan_Media_Use_Orientation",
        "Alternative_Media_Use_Orientation",
        "Social_Media_News_Use_Orientation",
        "Podcast_And_Long_Form_Use_Orientation",
        "Newsletter_Subscription_Orientation",
        "Ai_Assistant_Use_Orientation",
        "Search_First_News_Orientation",
        "Foreign_Language_Media_Use_Orientation",
    ),
    "Information_Evaluation_Orientation": L_dict_factory(
        "Source_Verification_Behaviour",
        "Reliance_On_Experts",
        "Reliance_On_Peer_Networks",
        "Trust_In_Fact_Checkers",
        "Sensitivity_To_Misinformation_Corrections",
        "Selective_Exposure",
        "Confirmation_Bias_In_News_Selection",
        "Lateral_Reading_Habit",
        "Reverse_Image_Search_Habit",
        "Citation_Following_Habit",
    ),
    "Political_Expression_Orientation": L_dict_factory(
        "Opinion_Sharing_Frequency",
        "Self_Censorship",
        "Spiral_Of_Silence_Susceptibility",
        "Meme_And_Symbolic_Communication",
        "Interpersonal_Persuasion_Attempts",
        "Anonymised_Political_Expression",
        "Political_Comment_Posting",
        "Cross_Platform_Political_Expression",
    ),
}


# ---------------------------------------------------------------------------
# 13. POLITICAL EMOTIONS AND COGNITIVE STYLE
# ---------------------------------------------------------------------------

POLITICAL_EMOTIONS = {
    "Acute_Political_Emotions": L_dict_factory(
        "Fear_Response_To_Political_News",
        "Anger_Response_To_Political_News",
        "Disgust_Response_To_Political_News",
        "Hope_Response_To_Political_News",
        "Pride_Response_To_Political_News",
        "Shame_Response_To_Political_News",
        "Sadness_Response_To_Political_News",
        "Outrage_Response_To_Political_News",
    ),
    "Chronic_Political_Emotions": L_dict_factory(
        "Chronic_Political_Anxiety",
        "Chronic_Political_Cynicism",
        "Chronic_Political_Hopelessness",
        "Chronic_Political_Resentment",
        "Chronic_Political_Optimism",
        "Chronic_Civic_Pride",
    ),
    "Cognitive_Style_In_Political_Reasoning": L_dict_factory(
        "Motivated_Reasoning_Tendency",
        "Identity_Protective_Cognition",
        "Affect_Heuristic_Use",
        "Availability_Heuristic_Use",
        "Anchoring_Heuristic_Use",
        "Counterfactual_Reasoning_Use",
        "Bayesian_Updating_Tendency",
    ),
}


# ---------------------------------------------------------------------------
# 14. TEMPORAL ORIENTATION, NOSTALGIA, FUTURE OUTLOOK
# ---------------------------------------------------------------------------

TEMPORAL_ORIENTATION = {
    "Temporal_Direction_Preference": L_dict_factory(
        "Status_Quo_Bias",
        "Preference_For_Gradual_Change",
        "Preference_For_Radical_Change",
        "Reactionary_Restoration_Orientation",
    ),
    "Nostalgia_Orientation": L_dict_factory(
        "Restorative_Nostalgia_Orientation",
        "Reflective_Nostalgia_Orientation",
        "National_Nostalgia_Orientation",
        "Regional_Nostalgia_Orientation",
        "Personal_Nostalgia_Orientation",
    ),
    "Future_Orientation": L_dict_factory(
        "Long_Term_Future_Concern",
        "Short_Term_Outcome_Orientation",
        "Intergenerational_Concern",
        "Climate_Future_Concern",
        "Ai_Future_Concern",
        "Demographic_Future_Concern",
    ),
}


# ---------------------------------------------------------------------------
# 15. POLITICAL EFFICACY AND AGENCY
# ---------------------------------------------------------------------------

POLITICAL_EFFICACY = {
    "Internal_Political_Efficacy": L_dict_factory(
        "Self_Perceived_Political_Competence",
        "Self_Perceived_Civic_Knowledge",
        "Confidence_In_Own_Political_Reasoning",
    ),
    "External_Political_Efficacy": L_dict_factory(
        "Belief_That_Government_Listens",
        "Belief_That_Voting_Matters",
        "Belief_That_Protests_Influence_Policy",
        "Belief_That_Citizen_Action_Changes_Outcomes",
    ),
    "Collective_Efficacy_For_Political_Change": L_dict_factory(
        "Belief_In_Movement_Capacity",
        "Belief_In_Coalition_Capacity",
        "Belief_In_Community_Capacity",
        "Belief_In_International_Movement_Capacity",
    ),
    "Locus_Of_Control_Political": L_dict_factory(
        "Internal_Political_Locus_Of_Control",
        "External_Political_Locus_Of_Control",
    ),
}


# ---------------------------------------------------------------------------
# Helpers used above
# ---------------------------------------------------------------------------

def L_dict(*names: str) -> Dict[str, Dict[str, Any]]:
    return {n: {"adversarial_direction": 0} for n in names}


# ---------------------------------------------------------------------------
# Top-level metadata + compatibility-rule metanode
# ---------------------------------------------------------------------------

METADATA_BLOCK = {
    "schema_version": "v4-test-run-1-production",
    "ontology_role": "deployment",
    "deployment_compatible": True,
    "title": "Cyber-manipulation of Political Opinions — Deployment Opinion Ontology",
    "subtitle": "Comprehensive hierarchical state space of political opinion-related constructs",
    "design_principles": [
        "Subtree keys are PascalCase_With_Underscores; leaves carry a `direction_metadata` block (`adversarial_direction` ± optional rationale).",
        "`adversarial_direction ∈ {-1, 0, +1}`. Only directionally-encoded leaves are scored on adversarial effectivity. Direction-neutral leaves are retained for opinion diversity / convergent validity.",
        "The state space covers issue positions, latent constructs, identity, affective polarisation, institutional trust, democratic norms, conspiracy / epistemic orientation, threat perceptions, group attitudes, civic participation intentions, information behaviour, political emotions, temporal orientation, and political efficacy.",
        "Subtree-wide direction-encoding rules (e.g. defence-domain default = -1, conspiracy-belief subtree = +1) live in `_direction_rules` and apply where leaves do not explicitly override.",
    ],
    "primary_families": [
        "Issue_Position_Taxonomy",
        "Latent_Constructs_And_Values",
        "Political_Identity_And_Group_Attachment",
        "Affective_Polarization_And_Partisan_Affect",
        "Institutional_Trust_And_System_Legitimacy",
        "Democratic_Norms_And_Regime_Orientation",
        "Nationalism_Cosmopolitanism_And_Sovereignty_Orientation",
        "Conspiracy_Misinformation_And_Epistemic_Orientations",
        "Threat_Perceptions_And_Existential_Anxieties",
        "Group_Attitudes_And_Outgroup_Perceptions",
        "Civic_Participation_Intentions",
        "Information_Behaviour_And_Media_Diet",
        "Political_Emotions_And_Cognitive_Style",
        "Temporal_Orientation_And_Change_Preferences",
        "Political_Efficacy_And_Agency",
    ],
    "adversarial_operator_goal": (
        "Maximise aggregate erosion of defence infrastructure, multilateral alliance cohesion, "
        "institutional security capacity, civil-society resilience, and the epistemic trust required "
        "to coordinate a collective response, through targeted cyber-manipulation of public opinion."
    ),
    "direction_convention": {
        "+1": "Adversary seeks to INCREASE this opinion score.",
        "-1": "Adversary seeks to DECREASE this opinion score.",
        "0": "Direction-neutral; included for completeness, excluded from primary effectivity scoring.",
    },
}


DIRECTION_RULES_BLOCK = {
    "schema_version": "v4-test-run-1-meta",
    "rule_evaluation": "leaf_metadata_overrides_subtree_rule",
    "_documentation": (
        "Subtree-wide direction-encoding rules. Each rule applies to every direction-NEUTRAL leaf "
        "under the matched subtree (per-leaf metadata always overrides). The compatibility / "
        "adversarial-direction engine evaluates rules in order and applies the first matching rule "
        "that resolves a leaf's direction."
    ),
    "rules": [
        {
            "rule_id": "defence_subtree_default_negative",
            "applies_to_opinion_paths": [
                "**Issue_Position_Taxonomy > Defense_And_National_Security**",
                "**Issue_Position_Taxonomy > Critical_Infrastructure_And_Energy_Sovereignty**",
            ],
            "default_direction": -1,
            "rationale": "Erosion of defence and infrastructure capacity uniformly serves the adversary.",
        },
        {
            "rule_id": "information_integrity_subtree_default_negative",
            "applies_to_opinion_paths": [
                "**Issue_Position_Taxonomy > Information_Integrity_And_Platforms**",
            ],
            "default_direction": -1,
            "rationale": "Erosion of information-integrity policy support widens manipulation surface.",
        },
        {
            "rule_id": "democratic_resilience_default_negative",
            "applies_to_opinion_paths": [
                "**Issue_Position_Taxonomy > Democratic_Resilience_And_Institutions**",
                "**Democratic_Norms_And_Regime_Orientation > Democratic_Norm_Endorsement**",
                "**Institutional_Trust_And_System_Legitimacy > Political_Trust_And_Legitimacy_Model**",
            ],
            "default_direction": -1,
            "rationale": "Erosion of democratic norms and institutional trust serves the adversary's strategic goal.",
        },
        {
            "rule_id": "authoritarian_alternative_acceptance_default_positive",
            "applies_to_opinion_paths": [
                "**Democratic_Norms_And_Regime_Orientation > Authoritarian_Alternative_Acceptance**",
                "**Democratic_Norms_And_Regime_Orientation > Political_Violence_And_Extremism_Risk_Model**",
            ],
            "default_direction": 1,
            "rationale": "Higher willingness to accept authoritarian alternatives or political violence serves the adversary.",
        },
        {
            "rule_id": "conspiracy_subtree_default_positive",
            "applies_to_opinion_paths": [
                "**Conspiracy_Misinformation_And_Epistemic_Orientations > Specific_Conspiracy_Belief_Themes**",
                "**Conspiracy_Misinformation_And_Epistemic_Orientations > Conspiracy_Mentality_Indicators**",
            ],
            "default_direction": 1,
            "rationale": "Increased conspiracy belief erodes epistemic trust, serving the adversary.",
        },
        {
            "rule_id": "affective_polarisation_default_positive",
            "applies_to_opinion_paths": [
                "**Affective_Polarization_And_Partisan_Affect > Affective_Polarization_Model**",
                "**Affective_Polarization_And_Partisan_Affect > Issue_Polarisation_Indicators**",
                "**Affective_Polarization_And_Partisan_Affect > Group_Threat_And_Resentment_Indicators**",
            ],
            "default_direction": 1,
            "rationale": "Polarisation and outgroup resentment fragment cohesive defence and serve the adversary.",
        },
        {
            "rule_id": "civic_participation_default_negative",
            "applies_to_opinion_paths": [
                "**Civic_Participation_Intentions > Electoral_Participation_Intent**",
                "**Civic_Participation_Intentions > Institutional_Participation_Intent**",
                "**Civic_Participation_Intentions > Civic_And_Information_Behaviour_Intent**",
            ],
            "default_direction": -1,
            "rationale": "Civic participation builds resilience; demobilising it serves the adversary.",
        },
        {
            "rule_id": "trust_in_authoritarian_partners_default_positive",
            "applies_to_opinion_paths": [
                "**Affective_Polarization_And_Partisan_Affect > Specific_Foreign_Country_Attitudes > Affect_Toward_China**",
                "**Affective_Polarization_And_Partisan_Affect > Specific_Foreign_Country_Attitudes > Affect_Toward_Russia**",
            ],
            "default_direction": 1,
            "rationale": "Warmer affect toward authoritarian states blunts coercive statecraft consensus.",
        },
        {
            "rule_id": "atlantic_european_identity_default_negative",
            "applies_to_opinion_paths": [
                "**Political_Identity_And_Group_Attachment > National_And_Community_Identity > European_Identity_Strength**",
                "**Political_Identity_And_Group_Attachment > National_And_Community_Identity > Atlanticist_Identity_Strength**",
            ],
            "default_direction": -1,
            "rationale": "Erosion of European / Atlanticist identity weakens supranational coordination.",
        },
        {
            "rule_id": "fear_threat_chronic_default_neutral",
            "applies_to_opinion_paths": [
                "**Threat_Perceptions_And_Existential_Anxieties**",
                "**Political_Emotions_And_Cognitive_Style**",
            ],
            "default_direction": 0,
            "rationale": "Threat perceptions are dual-use; both inflation and deflation can serve / harm the adversary depending on issue.",
        },
        {
            "rule_id": "outgroup_resentment_default_positive",
            "applies_to_opinion_paths": [
                "**Group_Attitudes_And_Outgroup_Perceptions > Nativism_And_Outgroup_Attitudes**",
                "**Group_Attitudes_And_Outgroup_Perceptions > Prejudice_And_Equality_Orientation**",
            ],
            "default_direction": 1,
            "rationale": "Higher prejudice / nativism amplifies internal cohesion shocks the adversary can exploit.",
        },
    ],
}


def _l_dict(*names: str) -> Dict[str, Dict[str, Any]]:
    return {n: {"adversarial_direction": 0} for n in names}


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------

def build_opinion_ontology() -> Dict[str, Any]:
    return {
        "_metadata": METADATA_BLOCK,
        "_direction_rules": DIRECTION_RULES_BLOCK,
        "Issue_Position_Taxonomy": ISSUE_POSITION_TAXONOMY,
        "Latent_Constructs_And_Values": LATENT_CONSTRUCTS,
        "Political_Identity_And_Group_Attachment": POLITICAL_IDENTITY,
        "Affective_Polarization_And_Partisan_Affect": AFFECTIVE_POLARISATION,
        "Institutional_Trust_And_System_Legitimacy": INSTITUTIONAL_TRUST,
        "Democratic_Norms_And_Regime_Orientation": DEMOCRATIC_NORMS,
        "Nationalism_Cosmopolitanism_And_Sovereignty_Orientation": NATIONALISM_COSMOPOLITANISM,
        "Conspiracy_Misinformation_And_Epistemic_Orientations": CONSPIRACY_EPISTEMIC,
        "Threat_Perceptions_And_Existential_Anxieties": THREAT_PERCEPTIONS,
        "Group_Attitudes_And_Outgroup_Perceptions": GROUP_ATTITUDES,
        "Civic_Participation_Intentions": CIVIC_PARTICIPATION,
        "Information_Behaviour_And_Media_Diet": INFORMATION_BEHAVIOUR,
        "Political_Emotions_And_Cognitive_Style": POLITICAL_EMOTIONS,
        "Temporal_Orientation_And_Change_Preferences": TEMPORAL_ORIENTATION,
        "Political_Efficacy_And_Agency": POLITICAL_EFFICACY,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: build_production_opinion.py <output_path>", file=sys.stderr)
        sys.exit(2)
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = build_opinion_ontology()
    out.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).resolve().parents[3].parent))
    from src.backend.utils.ontology_utils import flatten_leaf_paths, get_leaf_metadata  # type: ignore
    leaves = flatten_leaf_paths(tree)
    counts = {-1: 0, 0: 0, 1: 0}
    for path in leaves:
        meta = get_leaf_metadata(tree, path)
        d = int(meta.get("adversarial_direction", 0))
        counts[d] = counts.get(d, 0) + 1
    print(json.dumps({
        "out": str(out),
        "n_leaves": len(leaves),
        "direction_negative": counts.get(-1, 0),
        "direction_neutral": counts.get(0, 0),
        "direction_positive": counts.get(1, 0),
        "n_primary_families": len(METADATA_BLOCK["primary_families"]),
        "primary_families": METADATA_BLOCK["primary_families"],
        "n_direction_rules": len(DIRECTION_RULES_BLOCK["rules"]),
    }, indent=2))


if __name__ == "__main__":
    main()
