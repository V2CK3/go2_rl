// ============================================================================
// LCM_Bridge.cpp
// ----------------------------------------------------------------------------
// Go2 真机桥接程序：Unitree SDK2 (DDS)  <->  LCM  <->  Python 策略
//
// 【在整个部署链路中的位置】
//   Go2 电机/IMU/遥控器
//         │  DDS (rt/lowstate, rt/wirelesscontroller, rt/lowcmd)
//         ▼
//   本程序 LCM_Bridge          <—— 你现在看的文件
//         │  LCM 组播 (udpm://239.255.76.67:7667)
//         ▼
//   Python: StateBuilder / LCMAgent / deploy_go2_base.py
//
// 【三个工作线程】（默认 dt = 0.002s，即 500Hz）
//   1) lcm_send     : DDS LowState/Joystick -> 打包成 LCM 发出
//   2) lcm_receive  : 订阅 "pd_plustau_targets"，接收策略的关节 PD 目标
//   3) LowCmdWrite  : 把目标写入 LowCmd，经 DDS 发给电机；含 damping 安全逻辑
//
// 【LCM Topic 对照】
//   发出（C++ -> Python）:
//     "leg_control_data"      -> 12 关节 q / qd / tau_est
//     "state_estimator_data"  -> 姿态四元数 / RPY / 加速度 / 角速度 / 足力
//     "rc_command"            -> 摇杆与按键
//   接收（Python -> C++）:
//     "pd_plustau_targets"    -> q_des / qd_des / kp / kd / tau_ff
//
// 【用法】
//   cd sim2real_deploy/build
//   sudo ./LCM_Bridge eth0          # eth0 换成你的网卡名
//
// 【安全注意】
//   - 会关闭官方 sport_mode，进入 LOW-level 控制
//   - 启动前请吊起或让狗趴下
//   - 不要与 ./lcm_receive 同时运行
//   - 紧急：L2+B 进入 damping；再按 L2+B 退出；L2+A 恢复 sport_mode；L2+Y 回到策略控制
// ============================================================================

// -------------------- 头文件 --------------------
// LCM 消息类型（由 sim2real_deploy/lcm_types/*.lcm 经 lcm-gen 生成）
#include <lcm/lcm-cpp.hpp>
#include "leg_control_data_lcmt.hpp"
#include "state_estimator_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"

// 标准库
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cmath>

// Unitree SDK2：DDS 通道、底层状态/指令、遥控器、服务客户端、线程工具
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

// DDS Topic 名称（与 Unitree 官方约定一致，勿随意修改）
#define TOPIC_LOWCMD "rt/lowcmd"                 // 底层电机指令（本程序发布）
#define TOPIC_LOWSTATE "rt/lowstate"             // 底层状态（本程序订阅）
#define TOPIC_JOYSTICK "rt/wirelesscontroller"   // 无线遥控器（本程序订阅）

// 为保证项目代码的稳定性和易理解，没有采用 unitree_sdk2 中常用的 using namespace

// 位置/速度“停止占位值”：初始化 LowCmd 时使用，表示先不驱动到具体目标
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);


// ----------------------------------------------------------------------------
// CRC32：Unitree 底层指令帧校验。发送 LowCmd 前必须计算并写入 low_cmd.crc()
// 算法来自官方例程，一般无需修改。
// ----------------------------------------------------------------------------
uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}


// ----------------------------------------------------------------------------
// 遥控器按键位域：把 uint16 keys 拆成各个按键 bit
// 与 unitree_sdk2 例程一致，供安全逻辑 / 步态 mode 使用
// ----------------------------------------------------------------------------
typedef union
{
  struct
  {
    uint8_t R1 : 1;
    uint8_t L1 : 1;
    uint8_t start : 1;
    uint8_t select : 1;
    uint8_t R2 : 1;
    uint8_t L2 : 1;
    uint8_t F1 : 1;
    uint8_t F2 : 1;
    uint8_t A : 1;
    uint8_t B : 1;
    uint8_t X : 1;
    uint8_t Y : 1;
    uint8_t up : 1;
    uint8_t right : 1;
    uint8_t down : 1;
    uint8_t left : 1;
  } components;
  uint16_t value;
} xKeySwitchUnion;


// ============================================================================
// Custom：桥接核心类
//   - 持有 DDS 收发通道与 LCM 对象
//   - 在三个线程里完成 传感器转发 / 策略指令接收 / 电机写指令
// ============================================================================
class Custom
{
public:
    explicit Custom(){}
    ~Custom(){}

    // ---- 生命周期 ----
    void Init();                 // 订阅/发布通道初始化 + LCM 订阅策略指令
    void InitLowCmd();           // 把 LowCmd 填成安全的初始伺服帧
    void Loop();                 // 启动三个周期线程
    void SetNominalPose();       // 设置趴下/初始姿态的默认 PD 目标

    // ---- DDS 回调 ----
    void LowStateMessageHandler(const void* messages);  // 收到 LowState
    void JoystickHandler(const void *message);          // 收到遥控器

    // ---- 官方服务（开关 sport_mode 等）----
    void InitRobotStateClient();
    void activateService(const std::string& serviceName, int activate);
    int queryServiceStatus(const std::string& serviceName);

    // ---- 工作线程入口 ----
    void lcm_send();             // 线程1：状态 -> LCM
    void lcm_receive();          // 线程2：循环 handle LCM
    void lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf,
                             const std::string & chan,
                             const pd_tau_targets_lcmt* msg);  // 策略指令回调
    void LowCmdWrite();          // 线程3：写电机 + 安全状态机

    // -------------------- LCM 侧缓存（与 Python 交互的数据）--------------------
    leg_control_data_lcmt leg_control_lcm_data = {0};  // 发出：关节状态
    state_estimator_lcmt body_state_simple = {0};      // 发出：机身/IMU/足力
    pd_tau_targets_lcmt joint_command_simple = {0};    // 接收：策略 PD 目标
    rc_command_lcmt rc_command = {0};                  // 发出：遥控指令

    // -------------------- DDS 侧缓存 --------------------
    unitree_go::msg::dds_::LowState_ low_state{};              // 最新底层状态
    unitree_go::msg::dds_::LowCmd_ low_cmd{};                  // 待发送底层指令
    unitree_go::msg::dds_::WirelessController_ joystick{};     // 最新遥控数据
    unitree::robot::go2::RobotStateClient rsc;                 // 用于开关 sport_mode

    // DDS 通道指针
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_suber;

    // LCM 对象（默认组播；需与 Python 端 udpm://239.255.76.67:7667 一致）
    lcm::LCM lc;

    xKeySwitchUnion key;   // 当前按键位域
    int mode = 0;          // 步态/模式编号，由 A/B/X/Y/方向键写入，经 rc_command 传给 Python
    int motiontime = 0;    // LowCmdWrite 调用计数（可用于调试）
    float dt = 0.002;      // 线程周期 [s]，500Hz
    bool _firstRun;        // 首次写指令时，用当前关节角锁住目标，避免突变

    // 三个周期线程句柄
    unitree::common::ThreadPtr LcmSendThreadPtr;
    unitree::common::ThreadPtr LcmRecevThreadPtr;
    unitree::common::ThreadPtr lowCmdWriteThreadPtr;
};


// ============================================================================
// RobotStateClient：查询 / 开关 Go2 服务（本项目主要用于关闭 sport_mode）
// ============================================================================
void Custom::InitRobotStateClient()
{
    rsc.SetTimeout(5.0f);
    rsc.Init();
}

int Custom::queryServiceStatus(const std::string& serviceName)
{
    // 遍历服务列表，返回 1=已激活，0=未激活
    std::vector<unitree::robot::go2::ServiceState> serviceStateList;
    int ret, serviceStatus;
    ret = rsc.ServiceList(serviceStateList);
    size_t i, count = serviceStateList.size();
    for (i = 0; i < count; i++)
    {
        const unitree::robot::go2::ServiceState& serviceState = serviceStateList[i];
        if (serviceState.name == serviceName)
        {
            if (serviceState.status == 0)
            {
                std::cout << "name: " << serviceState.name <<" is activate"<<std::endl;
                serviceStatus = 1;
            }
            else
            {
                std::cout << "name:" << serviceState.name <<" is deactivate"<<std::endl;
                serviceStatus = 0;
            }
        }
    }
    return serviceStatus;
}

void Custom::activateService(const std::string& serviceName, int activate)
{
    // activate=0 关闭服务；activate=1 打开服务
    // 新版 SDK2: ServiceSwitch(name, swit, status)
    int32_t status = 0;
    rsc.ServiceSwitch(serviceName, activate, status);
}


// ============================================================================
// DDS 回调：只负责把最新消息拷到成员变量，真正的打包/写电机在周期线程里做
// ============================================================================
void Custom::LowStateMessageHandler(const void* message)
{
    // SDK2 回调传入的是 LowState_ 指针，保存整帧底层状态
    low_state = *(unitree_go::msg::dds_::LowState_*)message;
}

void Custom::JoystickHandler(const void *message)
{
    // 保存遥控摇杆，并把 keys 拆进位域联合体
    joystick = *(unitree_go::msg::dds_::WirelessController_ *)message;
    key.value = joystick.keys();
}


// ============================================================================
// 线程 1：lcm_send
// 作用：把最新 DDS LowState / Joystick 转成 LCM，供 Python StateBuilder 订阅
// 频率：约 500Hz（由 Loop() 里 CreateRecurrentThreadEx 的 dt 决定）
// ============================================================================
void Custom::lcm_send(){
    // ---- 1) 12 个电机状态 -> leg_control_data ----
    // SDK 电机顺序：通常 FR/FL/RR/RL 各 hip-thigh-calf；
    // Python 侧若需要训练顺序，会在 StateBuilder.joint_idxs 再重排。
    for (int i = 0; i < 12; i++)
    {
        leg_control_lcm_data.q[i] = low_state.motor_state()[i].q();           // 关节位置 [rad]
        leg_control_lcm_data.qd[i] = low_state.motor_state()[i].dq();         // 关节速度 [rad/s]
        leg_control_lcm_data.tau_est[i] = low_state.motor_state()[i].tau_est(); // 估计力矩 [Nm]
    }

    // ---- 2) IMU / 姿态 / 足力 -> state_estimator_data ----
    for (int i = 0; i < 4; i++){
        // 姿态四元数 wxyz（以 SDK 实际字段顺序为准）
        body_state_simple.quat[i] = low_state.imu_state().quaternion()[i];
    }
    for (int i = 0; i < 3; i++){
        body_state_simple.rpy[i] = low_state.imu_state().rpy()[i];              // roll/pitch/yaw
        body_state_simple.aBody[i] = low_state.imu_state().accelerometer()[i];  // 线加速度
        body_state_simple.omegaBody[i] = low_state.imu_state().gyroscope()[i];  // 角速度（注释里曾写“线性加速度”，实为陀螺仪）
    }
    for (int i = 0; i < 4; i++){
        // 足端力传感器读数，Python 里常用来估计接触
        body_state_simple.contact_estimate[i] = low_state.foot_force()[i];
    }

    // ---- 3) 遥控器 -> rc_command ----
    // 摇杆：lx/ly/rx/ry，范围约 [-1, 1]
    rc_command.left_stick[0] = joystick.lx();
    rc_command.left_stick[1] = joystick.ly();
    rc_command.right_stick[0] = joystick.rx();
    rc_command.right_stick[1] = joystick.ry();
    // 常用功能键映射到 LCM 字段（Python DeploymentRunner / StateBuilder 会读）
    rc_command.right_lower_right_switch = key.components.R2;  // 校准/启停常用
    rc_command.right_upper_switch = key.components.R1;
    rc_command.left_lower_left_switch = key.components.L2;     // 常与 B/A/Y 组合做安全操作
    rc_command.left_upper_switch = key.components.L1;

    // 面键 / 方向键：写入 mode，供 Python 切换步态（trot/bound/pace/...）
    if (key.components.A > 0){
        mode = 0;
    } else if (key.components.B > 0){
        mode = 1;
    } else if (key.components.X > 0){
        mode = 2;
    } else if (key.components.Y > 0){
        mode = 3;
    } else if (key.components.up > 0){
        mode = 4;
    } else if (key.components.right > 0){
        mode = 5;
    } else if (key.components.down > 0){
        mode = 6;
    } else if (key.components.left > 0){
        mode = 7;
    }
    rc_command.mode = mode;

    // ---- 4) 发布到 LCM（Python 订阅同名 topic）----
    lc.publish("leg_control_data", &leg_control_lcm_data);
    lc.publish("state_estimator_data", &body_state_simple);
    lc.publish("rc_command", &rc_command);
}


// ============================================================================
// 线程 2：lcm_receive + Handler
// 作用：接收 Python LCMAgent.publish_action() 发出的 pd_tau_targets
//
// 与 Python 的对应关系（见 sim2real_deploy/agent/lcm_agent.py）：
//   - 策略网络只直接输出期望关节增量；最终 q_des = default + action * scale
//   - kp / kd 来自训练配置 Cfg.control
//   - qd_des、tau_ff 通常为 0
// ============================================================================
void Custom::lcm_receive_Handler(const lcm::ReceiveBuffer *rbuf,
                                 const std::string & chan,
                                 const pd_tau_targets_lcmt* msg){
    (void) rbuf;
    (void) chan;
    // 整包覆盖本地缓存；真正写电机在 LowCmdWrite 里做
    joint_command_simple = *msg;
}

void Custom::lcm_receive(){
    // LCM 推荐写法：阻塞式 handle，有消息就回调到 lcm_receive_Handler
    // 注意：本函数本身是死循环；由独立线程调用，不会卡住其他线程
    while (true){
        lc.handle();
    }
}


// ============================================================================
// 线程 3 相关：LowCmd 初始化 / 默认姿态 / 写电机与安全状态机
// ============================================================================

void Custom::InitLowCmd()
{
    // 帧头与标志位：官方底层控制固定写法，用于 CRC 与协议识别
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;   // 底层控制标志
    low_cmd.gpio() = 0;

    // LowCmd 里有 20 个 motorCmd 槽位；Go2 实际只用前 12 个
    for (int i = 0; i < 20; i++)
    {
        // mode=0x01：伺服（PMSM）模式。若电机完全不动，优先检查这里是否被改掉
        low_cmd.motor_cmd()[i].mode() = (0x01);
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}

void Custom::SetNominalPose(){
    // 通信初始化后、Python 策略接管前：先给出一组“趴下/收腿”的安全 PD 目标
    // 注意：真正第一次有效写指令时，LowCmdWrite 会用当前关节角覆盖 q_des，避免突然跳变
    for (int i = 0; i < 12; i++){
        joint_command_simple.qd_des[i] = 0;
        joint_command_simple.tau_ff[i] = 0;
        joint_command_simple.kp[i] = 20;   // 初始化用较弱刚度
        joint_command_simple.kd[i] = 0.5;
    }

    // 预设趴下姿态（按 SDK 12 关节顺序）
    // 0-2 FR, 3-5 FL, 6-8 RR, 9-11 RL；每腿 hip / thigh / calf
    joint_command_simple.q_des[0] = -0.3;
    joint_command_simple.q_des[1] = 1.2;
    joint_command_simple.q_des[2] = -2.721;
    joint_command_simple.q_des[3] = 0.3;
    joint_command_simple.q_des[4] = 1.2;
    joint_command_simple.q_des[5] = -2.721;
    joint_command_simple.q_des[6] = -0.3;
    joint_command_simple.q_des[7] = 1.2;
    joint_command_simple.q_des[8] = -2.721;
    joint_command_simple.q_des[9] = 0.3;
    joint_command_simple.q_des[10] = 1.2;
    joint_command_simple.q_des[11] = -2.721;

    std::cout<<"SET NOMINAL POSE"<<std::endl;
}

void Custom::LowCmdWrite(){
    // 每个控制周期调用一次：决定本周期发给电机的 q/dq/kp/kd/tau
    motiontime ++;

    // ---- 首次有效状态到达：把目标锁到“当前角度”，防止从默认趴姿猛拉 ----
    if (_firstRun && leg_control_lcm_data.q[0] != 0){
        for (int i = 0; i < 12; i++){
            joint_command_simple.q_des[i] = leg_control_lcm_data.q[i];
            // 清掉可能残留的组合键状态，避免一启动就进 damping
            key.components.Y = 0;
            key.components.A = 0;
            key.components.B = 0;
            key.components.L2 = 0;
        }
        _firstRun = false;
    }

    // ---- 安全判定：姿态过大 或 按下 L2+B -> 进入 damping ----
    // roll / pitch 阈值 0.8 rad（约 46°）。也可按需要改严/改松。
    if (std::abs(low_state.imu_state().rpy()[0]) > 0.8 ||
        std::abs(low_state.imu_state().rpy()[1]) > 0.8 ||
        ((int)key.components.B == 1 && (int)key.components.L2 == 1))
    {
        // damping：kp=0, kd>0，电机被动阻尼，不主动跟位置
        for (int i = 0; i < 12; i++){
            low_cmd.motor_cmd()[i].q() = 0;
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 0;
            low_cmd.motor_cmd()[i].kd() = 5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }
        std::cout << "======= Switched to Damping Mode, and the thread is sleeping ========"<<std::endl;
        sleep(1.5);

        // damping 后的人工决策环（阻塞本写指令线程，直到用户选一项）
        while (true)
        {
            if (((int)key.components.B == 1 && (int)key.components.L2 == 1)) {
                // 再次 L2+B：直接退出进程
                std::cout << "======= [L2+B] is pressed again, the script is about to exit========" <<std::endl;
                exit(0);
            } else if (((int)key.components.A == 1 && (int)key.components.L2 == 1)){
                // L2+A：恢复官方 sport_mode，然后退出
                int32_t status = 0;
                rsc.ServiceSwitch("sport_mode", 1, status);
                std::cout << "======= activate sport_mode service and exit========" <<std::endl;
                sleep(0.5);
                exit(0);
            } else{
                if (((int)key.components.Y == 1 && (int)key.components.L2 == 1)){
                    // L2+Y：退出 damping，回到 Python RL 策略控制
                    std::cout << "=======  Switch to RL policy ========"<<std::endl;
                    std::cout<<"Communicatino is set up successfully" << std::endl;
                    std::cout<<"LCM <<<------------>>> Unitree SDK2" << std::endl;
                    std::cout<<"------------------------------------" << std::endl;
                    std::cout<<"------------------------------------" << std::endl;
                    std::cout<<"Press L2+B if any unexpected error occurs" << std::endl;
                    break;
                } else{
                    std::cout << "======= Press [L2+B] again to exit ========"<<std::endl;
                    std::cout << "======= Press [L2+Y] again to switch to RL policy ========"<<std::endl;
                    std::cout << "======= Press [L2+A] again to activate sport_mode service========"<<std::endl;
                    sleep(0.01);
                }
            }
        }
    }
    else{
        // ---- 正常路径：执行策略（或初始化）下发的 PD 目标 ----
        for (int i = 0; i < 12; i++){
            low_cmd.motor_cmd()[i].q() = joint_command_simple.q_des[i];
            low_cmd.motor_cmd()[i].dq() = joint_command_simple.qd_des[i];
            low_cmd.motor_cmd()[i].kp() = joint_command_simple.kp[i];
            low_cmd.motor_cmd()[i].kd() = joint_command_simple.kd[i];
            low_cmd.motor_cmd()[i].tau() = joint_command_simple.tau_ff[i];
        }
    }

    // 计算 CRC 并经 DDS 发出；电机驱动板收到后按 PD 跟踪
    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    lowcmd_publisher->Write(low_cmd);
}


// ============================================================================
// Init / Loop / main：启动流程
// ============================================================================

void Custom::Init(){
    _firstRun = true;
    InitLowCmd();
    SetNominalPose();

    // 订阅 Python 策略输出的关节目标
    // topic 名必须与 lcm_agent.py 里 lc.publish("pd_plustau_targets", ...) 一致
    lc.subscribe("pd_plustau_targets", &Custom::lcm_receive_Handler, this);

    // 创建 DDS：发 LowCmd、收 LowState、收 Joystick
    lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);

    joystick_suber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    joystick_suber->InitChannel(std::bind(&Custom::JoystickHandler, this, std::placeholders::_1), 1);
}


void Custom::Loop(){
    // CreateRecurrentThreadEx(name, cpu, interval_us, fn, this)
    // dt=0.002s => interval = 2000 us
    LcmSendThreadPtr = unitree::common::CreateRecurrentThreadEx(
        "lcm_send_thread", UT_CPU_ID_NONE, dt * 1e6, &Custom::lcm_send, this);
    LcmRecevThreadPtr = unitree::common::CreateRecurrentThreadEx(
        "lcm_recev_thread", UT_CPU_ID_NONE, dt * 1e6, &Custom::lcm_receive, this);
    lowCmdWriteThreadPtr = unitree::common::CreateRecurrentThreadEx(
        "dds_write_thread", UT_CPU_ID_NONE, dt * 1e6, &Custom::LowCmdWrite, this);
}

int main(int argc, char **argv)
{
    // 必须传入网卡名，例如 eth0 / enp3s0；SDK2 用它绑定 DDS 通信
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }

    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Caution: The scripts is about to shutdown Unitree sport_mode Service." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    // 初始化 DDS 通道工厂：参数 0=域，argv[1]=网卡接口名（PC 或 Jetson）
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    Custom custom;

    // 进入 LOW-level 前必须关闭官方运动服务，否则底层指令会被 sport_mode 抢占
    custom.InitRobotStateClient();
    if (custom.queryServiceStatus("sport_mode"))
    {
        std::cout<<"Trying to deactivate the service: " << "sport_mode" << std::endl;
        custom.activateService("sport_mode", 0);
        sleep(0.5);
        if (!custom.queryServiceStatus("sport_mode")){
            std::cout<<"Trying to deactivate the service: " << "sport_mode" << std::endl;
        }
    } else{
        std::cout <<"sportd_mode is already deactivated now" << std::endl
                  <<"next step is setting up communication" << std::endl
                  << "Press Enter to continue..." << std::endl;
        std::cin.ignore();
    }

    // 建立 DDS/LCM 订阅发布，并给出初始姿态目标
    custom.Init();

    std::cout<<"Communicatino is set up successfully" << std::endl;
    std::cout<<"LCM <<<------------>>> Unitree SDK2" << std::endl;
    std::cout<<"------------------------------------" << std::endl;
    std::cout<<"------------------------------------" << std::endl;
    std::cout<<"Press L2+B if any unexpected error occurs" << std::endl;

    // 启动三线程后，主线程休眠保活
    custom.Loop();

    while (true)
    {
        sleep(10);
    }

    return 0;
}
