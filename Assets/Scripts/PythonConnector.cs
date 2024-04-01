using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class PythonConnector : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5065; // Select a port that fits your setup

    void Start()
    {
        InitializeUDP();
    }

    private void InitializeUDP()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);
                byte[] data = client.Receive(ref anyIP);

                string text = Encoding.UTF8.GetString(data);
                Debug.Log(">> " + text);
            }
            catch (Exception e)
            {
                Debug.Log(e.ToString());
            }
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        client.Close();
    }
}