import csv
import os

def make_ppo_losses_csv(save_path="./"):
    file = os.path.join(save_path, 'losses.csv')
    with open(file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['timestamp', 'sps', 'step', 'learning_rate', 'value_loss', 'policy_loss', 'entropy',
                                'old_approx_kl', 'approx_kl', 'clipfrac', 'explained_variance', 'bc_loss', 'bc_accuracy'])
    return file

def make_pqn_losses_csv(save_path="./"):
    file = os.path.join(save_path, 'losses.csv')
    with open(file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['timestamp', 'sps', 'step', 'learning_rate', 'td_loss'])
    return file

def make_training_csv_craftax_classic(save_path="./"):
    file = os.path.join(save_path, 'training.csv')
    with open(file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['timestamp', 'sps', 'step', 'avg_ep_reward', 'avg_ep_length', 'collect_coal',
                                'collect_diamond', 'collect_drink', 'collect_iron',
                                'collect_sapling', 'collect_stone', 'collect_wood',
                                'defeat_skeleton', 'defeat_zombie', 'eat_cow',
                                'eat_plant', 'make_iron_pickaxe', 'make_iron_sword',
                                'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe',
                                'make_wood_sword', 'place_furnace', 'place_plant',
                                'place_stone', 'place_table', 'wake_up'])
    return file

def make_training_csv_craftax(save_path="./"):
    file = os.path.join(save_path, 'training.csv')
    with open(file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['timestamp', 'sps', 'step', 'avg_ep_reward', 'avg_ep_length', 'cast_fireball', 'cast_iceball',
                             'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
                             'collect_ruby', 'collect_sapling', 'collect_sapphire', 'collect_stone',
                             'collect_wood', 'damage_necromancer', 'defeat_archer', 'defeat_deep_thing',
                             'defeat_fire_elemental', 'defeat_frost_troll', 'defeat_gnome_archer', 'defeat_gnome_warrior',
                             'defeat_ice_elemental', 'defeat_knight', 'defeat_kobold', 'defeat_lizard', 'defeat_necromancer',
                             'defeat_orc_mage', 'defeat_orc_solider', 'defeat_pigman', 'defeat_skeleton', 'defeat_troll',
                             'defeat_zombie', 'drink_potion', 'eat_bat', 'eat_cow', 'eat_plant', 'eat_snail',
                             'enchant_armour', 'enchant_sword', 'enter_dungeon', 'enter_fire_realm', 'enter_gnomish_mines',
                             'enter_graveyard', 'enter_ice_realm', 'enter_sewers', 'enter_troll_mines', 'enter_vault',
                             'find_bow', 'fire_bow', 'learn_fireball', 'learn_iceball', 'make_arrow',
                             'make_diamond_armour', 'make_diamond_pickaxe', 'make_diamond_sword', 'make_iron_armour',
                             'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
                             'make_torch', 'make_wood_pickaxe', 'make_wood_sword', 'open_chest', 'place_furnace',
                             'place_plant', 'place_stone', 'place_table', 'place_torch', 'wake_up'])
    return file
    
def write_row_csv(file, row):
    with open(file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)


def save_results_csv(args, algo_name, avg_ep_reward, avg_ep_length, avg_achievements, total_steps, wall_time_seconds):
    """Save results CSV in the {hypothesis_id}__{experiment_id}.csv convention."""
    if not args.hypothesis_id or not args.experiment_id:
        return

    output_dir = args.output_dir if args.output_dir else "."
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{args.hypothesis_id}__{args.experiment_id}.csv"
    filepath = os.path.join(output_dir, filename)

    import numpy as np

    row = {
        "hypothesis_id": args.hypothesis_id,
        "experiment_id": args.experiment_id,
        "algorithm": algo_name,
        "seed": args.seed,
        "total_timesteps": total_steps,
        "wall_time_seconds": wall_time_seconds,
        "avg_episode_return": np.mean(avg_ep_reward) if len(avg_ep_reward) > 0 else 0,
        "avg_episode_length": np.mean(avg_ep_length) if len(avg_ep_length) > 0 else 0,
    }
    for k, v in avg_achievements.items():
        clean_key = k.replace("Achievements/", "")
        row[clean_key] = np.mean(v) if len(v) > 0 else 0

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"Results saved to {filepath}")