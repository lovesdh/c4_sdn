package org.onosproject.ngsdn.tutorial;

import com.google.common.collect.Lists;
import org.onlab.packet.Ip6Address;
import org.onlab.packet.Ip6Prefix;
import org.onlab.packet.IpAddress;
import org.onlab.packet.IpPrefix;
import org.onlab.packet.MacAddress;
import org.onlab.util.ItemNotFoundException;
import org.onosproject.core.ApplicationId;
import org.onosproject.mastership.MastershipService;
import org.onosproject.net.Device;
import org.onosproject.net.DeviceId;
import org.onosproject.net.Host;
import org.onosproject.net.Link;
import org.onosproject.net.PortNumber;
import org.onosproject.net.config.NetworkConfigService;
import org.onosproject.net.device.DeviceEvent;
import org.onosproject.net.device.DeviceListener;
import org.onosproject.net.device.DeviceService;
import org.onosproject.net.flow.FlowRule;
import org.onosproject.net.flow.FlowRuleService;
import org.onosproject.net.flow.criteria.PiCriterion;
import org.onosproject.net.group.GroupDescription;
import org.onosproject.net.group.GroupService;
import org.onosproject.net.host.HostEvent;
import org.onosproject.net.host.HostListener;
import org.onosproject.net.host.HostService;
import org.onosproject.net.host.InterfaceIpAddress;
import org.onosproject.net.intf.Interface;
import org.onosproject.net.intf.InterfaceService;
import org.onosproject.net.link.LinkEvent;
import org.onosproject.net.link.LinkListener;
import org.onosproject.net.link.LinkService;
import org.onosproject.net.pi.model.PiActionId;
import org.onosproject.net.pi.model.PiActionParamId;
import org.onosproject.net.pi.model.PiMatchFieldId;
import org.onosproject.net.pi.runtime.PiAction;
import org.onosproject.net.pi.runtime.PiActionParam;
import org.onosproject.net.pi.runtime.PiActionProfileGroupId;
import org.onosproject.net.pi.runtime.PiTableAction;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.onosproject.ngsdn.tutorial.common.FabricDeviceConfig;
import org.onosproject.ngsdn.tutorial.common.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.onlab.util.ImmutableByteSequence;


import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import static com.google.common.collect.Streams.stream;
import static org.onosproject.ngsdn.tutorial.AppConstants.INITIAL_SETUP_DELAY;

import org.onlab.util.ImmutableByteSequence;
import org.onosproject.net.flow.criteria.PiCriterion.Builder;

/**
 * App component that configures devices to provide IPv6 routing capabilities
 * across the whole fabric.
 */
@Component(
        immediate = true,
        enabled = true
)


public class GuideComponent {

    private class Dst{
        public Ip6Address ip;
        public MacAddress mac;
        public Dst(Ip6Address ip,MacAddress mac){
            this.ip = ip;
            this.mac = mac;
        }
    }
    private static final Logger log = LoggerFactory.getLogger(GuideComponent.class);

    private static final int DEFAULT_GUIDE_GROUP_ID = 0xec3a0000;
    private static final long GROUP_INSERT_DELAY_MILLIS = 200;  //延迟时间
    private static final int PACKET_CLONE_SESSION_ID = 100;

    private final DeviceListener deviceListener = new InternalDeviceListener();

    private ApplicationId appId;


    //--------------------------------------------------------------------------
    // ONOS CORE SERVICE BINDING
    //
    // These variables are set by the Karaf runtime environment before calling
    // the activate() method.
    //--------------------------------------------------------------------------

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private FlowRuleService flowRuleService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private HostService hostService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private MastershipService mastershipService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private GroupService groupService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private DeviceService deviceService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private NetworkConfigService networkConfigService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private InterfaceService interfaceService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private LinkService linkService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    private MainComponent mainComponent;

    //--------------------------------------------------------------------------
    // COMPONENT ACTIVATION.
    //
    // When loading/unloading the app the Karaf runtime environment will call
    // activate()/deactivate().
    //--------------------------------------------------------------------------

    @Activate
    protected void activate() {
        appId = mainComponent.getAppId();

        // hostService.addListener(hostListener);
        // linkService.addListener(linkListener);
        deviceService.addListener(deviceListener);

        // Schedule set up for all devices.
        // mainComponent.scheduleTask(this::setUpAllDevices, INITIAL_SETUP_DELAY);

        log.info("Started");
    }

    @Deactivate
    protected void deactivate() {
        // hostService.removeListener(hostListener);
        // linkService.removeListener(linkListener);
        deviceService.removeListener(deviceListener);

        log.info("Stopped");
    }

    /**
     * Listener of device events which triggers configuration of the My Station
     * table.
     */
    class InternalDeviceListener implements DeviceListener {

        @Override
        public boolean isRelevant(DeviceEvent event) {
            switch (event.type()) {
                case DEVICE_AVAILABILITY_CHANGED:
                case DEVICE_ADDED:
                    break;
                default:
                    return false;
            }
            // Process device event if this controller instance is the master
            // for the device and the device is available.
            DeviceId deviceId = event.subject().id();
            return mastershipService.isLocalMaster(deviceId) &&
                    deviceService.isAvailable(event.subject().id());
        }

        @Override
        public void event(DeviceEvent event) {
            mainComponent.getExecutorService().execute(() -> {
                DeviceId deviceId = event.subject().id();
                log.info("{} event! device id={}", event.type(), deviceId);
                if(deviceId.toString().equals("device:leaf1")){
                    setUpGuideTable(deviceId);
                    setUpSelectHostTable(deviceId);
                }
                
            });
        }
    }


    //创建检测host的group,定义三个组成员
    private GroupDescription createGuideHostGroup(int groupId,
                                                Collection<Ip6Address> hostIps,                                    
                                                DeviceId deviceId) {

        String actionProfileId = "EgressPipeImpl.host_selector";

        Ip6Address firstIp = Ip6Address.valueOf("3:103:2::");
        Ip6Address secIp = Ip6Address.valueOf("3:204:2::");

        final List<PiAction> actions = Lists.newArrayList();

        final String tableId = "EgressPipeImpl.select_host";

        for(Ip6Address hostip:hostIps){
            final PiAction action = PiAction.builder()
                     .withId(PiActionId.of("EgressPipeImpl.srv6_t_insert_3"))
                     .withParameter(new PiActionParam(PiActionParamId.of("s1"), firstIp.toOctets()))
                     .withParameter(new PiActionParam(PiActionParamId.of("s2"), secIp.toOctets()))
                     .withParameter(new PiActionParam(PiActionParamId.of("s3"), hostip.toOctets()))
                     .build();
            
            actions.add(action);
        }

        return Utils.buildSelectGroup(deviceId,tableId,actionProfileId,groupId,actions,appId);

    }

    //下发guide table的表项
    private void setUpGuideTable(DeviceId deviceId) {
        log.info("Setting up guide table for device {}", deviceId);

        final String tableId = "IngressPipeImpl.guide_table";

        
        final PiCriterion match = PiCriterion.builder()
                .matchTernary(PiMatchFieldId.of("hdr.ipv6.src_addr"), Ip6Address.valueOf("2001:1:1::0").toOctets(), Ip6Address.valueOf("FFFF:FFFF:FFFF::0").toOctets())
                .build();
    
        final PiTableAction action = PiAction.builder()
                .withId(PiActionId.of("IngressPipeImpl.packet_clone"))
                .build();

        final FlowRule guideRule = Utils.buildFlowRule(deviceId,appId,tableId,match,action);

        flowRuleService.applyFlowRules(guideRule);
    }


    //下发选择检测host的表项
    private void setUpSelectHostTable(DeviceId deviceId) {
        log.info("Setting up select host table for device {}", deviceId);

        
        final int groupId = 1;

        // Dst dst1 = new Dst(Ip6Address.valueOf("2001:1:4::a"),MacAddress.valueOf("00:00:00:00:00:4A"));
        // Dst dst2 = new Dst(Ip6Address.valueOf("2001:1:4::b"),MacAddress.valueOf("00:00:00:00:00:4B"));
        // Dst dst3 = new Dst(Ip6Address.valueOf("2001:1:4::c"),MacAddress.valueOf("00:00:00:00:00:4C"));

        // List<Dst> dsts = Lists.newArrayList();
        // dsts.add(dst1);
        // dsts.add(dst2);
        // dsts.add(dst3);


        List<Ip6Address> ips = Lists.newArrayList();
        ips.add(Ip6Address.valueOf("2001:1:4::a"));
        ips.add(Ip6Address.valueOf("2001:1:4::b"));
        ips.add(Ip6Address.valueOf("2001:1:4::c"));


        // GroupDescription group = createGuideHostGroup(groupId,ips,deviceId,appId);
        // groupService.addGroup(group);

        int is_clone = 1;

        final GroupDescription group = createGuideHostGroup(groupId,ips,deviceId);

        final GroupDescription cloneGroup = Utils.buildCloneGroup(
                appId,
                deviceId,
                PACKET_CLONE_SESSION_ID,
                // Ports where to clone the packet.
                // Just controller in this case.
                Collections.singleton(PortNumber.portNumber(2)));

        final String tableId = "EgressPipeImpl.select_host";
        final PiCriterion match = PiCriterion.builder()
                .matchExact(
                        PiMatchFieldId.of("local_metadata.hit_bit"),
                        is_clone)
                .build();

        final PiTableAction action = PiActionProfileGroupId.of(groupId);

        final FlowRule rule =  Utils.buildFlowRule(deviceId,appId,tableId,match,action);

        try{
            groupService.addGroup(group);
            groupService.addGroup(cloneGroup);
            Thread.sleep(GROUP_INSERT_DELAY_MILLIS);
            flowRuleService.applyFlowRules(rule);
        }catch(InterruptedException e){
            log.error("Interrupted!",e);
            Thread.currentThread().interrupt();
        }

    }

    
}