// ============================================================================
// receive_msgs_test.cpp
// ----------------------------------------------------------------------------
// LCM 通信自检工具（只读、不写电机）
//
// 【用途】
//   在启动正式桥接程序 lcm_position_go2 之前，验证：
//     1) 本机 LCM 能否正常工作
//     2) 是否能收到由桥接程序（或其它发布者）发出的三类状态消息
//
// 【重要】
//   - 本程序只订阅并打印，不会向机器人发 LowCmd
//   - 正式部署时：先用本程序验证，确认后关掉它，再运行 lcm_position_go2
//   - 不要与 lcm_position_go2 长时间并行抢同一调试环境（README 明确要求不要同时跑）
//
// 【订阅的 LCM Topic】（应由 lcm_position_go2::lcm_send 发布）
//   "leg_control_data"      -> 12 关节角度/速度/估计力矩
//   "state_estimator_data"  -> 四元数 / RPY / IMU / 足力
//   "rc_command"            -> 遥控器摇杆与按键
//
// 【用法】
//   # 终端 1：先开桥（发布端）
//   sudo ./lcm_bridge <网卡名>
//   # 终端 2：自检（只读）
//   ./lcm_receive_msgs_test
// ============================================================================

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <lcm/lcm-cpp.hpp>

// 以下头文件由 lcm-gen 从 sim2real_deploy/lcm_types/*.lcm 自动生成
#include "leg_control_data_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "rc_command_lcmt.hpp"


// ============================================================================
// Handler1：打印腿部控制数据
// topic: "leg_control_data"
// 字段含义与 leg_control_data_lcmt.lcm 一致：
//   q[12]       关节位置 [rad]
//   qd[12]      关节速度 [rad/s]
//   tau_est[12] 估计力矩 [Nm]
// 关节下标通常对应 SDK 顺序（FR/FL/RR/RL 各 3 个关节）
// ============================================================================
class Handler1
{
    public:
        ~Handler1() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan,
                const leg_control_data_lcmt* leg_control_lcm_data)
        {
            (void)rbuf;
            (void)chan;

            int tab_space = 15; // 表格列宽，方便终端对齐阅读
            std::cout << "**************** msgs name : leg_control_lcm_data ****************" << std::endl;
            std::cout << std::left
                      << std::setw(tab_space) << "Motor id"
                      << std::setw(tab_space) << "angle"
                      << std::setw(tab_space) << "velocity"
                      << std::setw(tab_space) << "torque"
                      << std::endl;

            // 逐电机打印：若数值长期全 0，多半是发布端未启动或 DDS/网卡未通
            for (int i = 0; i < 12; i++){
                std::cout << std::left
                          << std::setw(tab_space) << i
                          << std::setw(tab_space) << leg_control_lcm_data->q[i]
                          << std::setw(tab_space) << leg_control_lcm_data->qd[i]
                          << std::setw(tab_space) << leg_control_lcm_data->tau_est[i]
                          << std::endl;
            }
            std::cout << std::endl;
        }
};


// ============================================================================
// Handler2：打印机身状态估计 / IMU / 足力
// topic: "state_estimator_data"
// 对应 state_estimator_lcmt.lcm：
//   quat[4]              姿态四元数
//   rpy[3]               roll / pitch / yaw
//   aBody[3]             机体系加速度
//   omegaBody[3]         机体系角速度
//   contact_estimate[4]  四足足力（桥接里直接填 foot_force）
// ============================================================================
class Handler2
{
    public:
        ~Handler2() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan,
                const state_estimator_lcmt* state_estimator_data)
        {
            (void)rbuf;
            (void)chan;

            std::cout << "**************** msgs name: state_estimator_data ****************" << std::endl;

            // 姿态四元数：用于检查 IMU 是否在更新
            std::cout << "quaternion: " << state_estimator_data->quat[0] << '\t'
                                        << state_estimator_data->quat[1] << '\t'
                                        << state_estimator_data->quat[2] << '\t'
                                        << state_estimator_data->quat[3] << '\t'<< std::endl;

            // RPY：真机安全逻辑也会看 roll/pitch 是否过大
            std::cout << "posture angles: " << std::endl
                                            << "roll: "<< state_estimator_data->rpy[0] << '\t'
                                            << "pitch: "<< state_estimator_data->rpy[1] << '\t'
                                            << "yaw: "<< state_estimator_data->rpy[2] << '\t'<< std::endl;

            // 线加速度（机体系）
            std::cout << "imu acc: " << std::endl
                                            << "ax: "<< state_estimator_data->aBody[0] << '\t'
                                            << "ay: "<< state_estimator_data->aBody[1] << '\t'
                                            << "az: "<< state_estimator_data->aBody[2] << '\t'<< std::endl;

            // 角速度（机体系）；下方 "wa" 实为 wz，沿用原打印标签
            std::cout << "imu omega: " << std::endl
                                            << "wx: "<< state_estimator_data->omegaBody[0] << '\t'
                                            << "wy: "<< state_estimator_data->omegaBody[1] << '\t'
                                            << "wa: "<< state_estimator_data->omegaBody[2] << '\t'<< std::endl;

            // 足力：数值随站立/抬腿变化；全 0 可能是未站立或传感器未通
            std::cout << "foot force:  " << std::endl
                                            << "FR foot force: "<< state_estimator_data->contact_estimate[0] << std::endl
                                            << "FL foot force: "<< state_estimator_data->contact_estimate[1] << std::endl
                                            << "RR foot force: "<< state_estimator_data->contact_estimate[2] << std::endl
                                            << "RL foot force: "<< state_estimator_data->contact_estimate[3] << std::endl<< std::endl;
        }
};


// ============================================================================
// Handler3：打印遥控器数据
// 期望对应 rc_command_lcmt.lcm：
//   left_stick[2] / right_stick[2]
//   L1/L2/R1/R2 等开关量
//
// 注意订阅 topic 名：这里是 "rc_command_data"
// 而 lcm_position_go2.cpp 发布的是 "rc_command"
// 若只有腿部/IMU 有输出、遥控一直无输出，就是 topic 不一致导致
// ============================================================================
class Handler3
{
    public:
        ~Handler3() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan,
                const rc_command_lcmt* rc_command_data)
        {
            (void)rbuf;
            (void)chan;

            std::cout << "**************** msgs name: rc_command_data ****************" << std::endl;
            std::cout << "lx:  " << rc_command_data->left_stick[0] << std::endl;   // 左摇杆 X
            std::cout << "ly:  " << rc_command_data->left_stick[1] << std::endl;   // 左摇杆 Y
            std::cout << "rx:  " << rc_command_data->right_stick[0] << std::endl;  // 右摇杆 X
            std::cout << "ry:  " << rc_command_data->right_stick[1] << std::endl;  // 右摇杆 Y
            std::cout << "R1:  " << rc_command_data->right_upper_switch << std::endl;
            std::cout << "R2:  " << rc_command_data->right_lower_right_switch << std::endl;
            std::cout << "L1:  " << rc_command_data->left_upper_switch << std::endl;
            std::cout << "L2:  " << rc_command_data->left_lower_left_switch << std::endl;
            std::cout << "------------------------------------------------------------------------------------------" << std::endl;
        }
};


// ============================================================================
// main：创建 LCM、注册三个 Handler，然后阻塞等待消息
// ============================================================================
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    // 使用默认 LCM URL（通常为组播）。需与发布端一致。
    lcm::LCM lc;
    if (!lc.good()){
        // 常见原因：LCM 未正确安装、权限不足、组播被禁
        std::cout << "lcm is error" << std::endl;
        return 1;
    }

    Handler1 handlerObject_leg_control_data;
    Handler2 handlerObject_state_estimator;
    Handler3 handlerObject_rc_command;

    // 绑定 topic -> 回调。消息一到就会打印。
    lc.subscribe("leg_control_data", &Handler1::handleMessage, &handlerObject_leg_control_data);
    lc.subscribe("state_estimator_data", &Handler2::handleMessage, &handlerObject_state_estimator);
    lc.subscribe("rc_command", &Handler3::handleMessage, &handlerObject_rc_command);

    // lc.handle()：阻塞等待并分发一条消息
    // 返回 0 表示成功处理；返回 -1 表示出错（例如 LCM 失效）
    // 成功时循环继续；出错时跳出并结束进程
    while (0 == lc.handle()){
        // 所有业务都在 Handler 回调里完成，这里无需额外处理
    };

    return 0;
}
