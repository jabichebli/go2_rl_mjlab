#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
// EDITED: You may need to include the Unitree common types for the remote struct if not already linked
#include "unitree/common/decl.hpp" 

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    // --- EDITED: HEIGHT COMMAND INJECTION ---
    // 1. Parse the raw remote control data from the Unitree SDK2 LowState message
    unitree::common::xRockerBtnDataStruct remote;
    memcpy(&remote, FSMState::lowstate->msg_.wireless_remote().data(), sizeof(remote));
    
    // 2. Map remote.ry (Right Joystick Up/Down) from [-1.0, 1.0] to our height range [0.15, 0.38]
    // Midpoint is 0.265, Half-range is 0.115
    float target_height = 0.265 + (remote.ry * 0.115);

    // 3. Inject this target height into the 4th index (index 3) of the twist command tensor
    // This makes sure the C++ observation manager sees the height command you just requested
    if (env->command_manager->commands.find("base_velocity") != env->command_manager->commands.end()) {
        env->command_manager->commands["base_velocity"].data_ptr<float>()[3] = target_height;
    }
    // ----------------------------------------

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
