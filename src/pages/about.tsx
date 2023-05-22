// @VisualDust 2023 all rights reserved
import React, { useEffect, useState } from "react";
import Layout from "@theme/Layout";
import clsx from "clsx";
import arrayShuffle from "array-shuffle";

function About() {
  return (
    <Layout>
      <Friends />
    </Layout>
  );
}

interface FriendData {
  pic: string;
  name: string;
  intro: string;
  url: string;
  note: string;
  ip: string;
}

function githubPic(name: string) {
  return `https://avatars.githubusercontent.com/u/${name}?v=4`;
}
var friendsData: FriendData[] = [
  {
    pic: githubPic("56100984"),
    name: "Zhiying Liang",
    intro: "深度潜水选手",
    url: "http://joyceliang.club/",
    note: "BrickBar首位搬砖人，会一点但不多，人菜瘾大，在大浪潮里荡啊荡～ 目前研究方向主要是智能多模态融合(超声钼靶balabala之类的医学影像",
    ip:"Graduate Student, Biomedical Engineering School, ShanghaiTech University",
  },
  {
    pic: githubPic("62140756"),
    name: "Yunhao Li",
    intro: "刚入门炼丹，正在努力炼出一炉好丹药",
    url: "https://peterli.club/",
    note: "会的都会，不会的一学就会，问就是啥都会的全栈选手，被学院同级誉为最猛的全能神？目前研究方向主要是Computer Vision, Segmentation and Generated models",
    ip: "Graduate Student, AI School, Guangzhou University",

  },
];

function Friends() {
  const [friends, setFriends] = useState<FriendData[]>(friendsData);
  useEffect(() => {
    setFriends(arrayShuffle(friends))
  }, []);
  const [current, setCurrent] = useState(0);
  const [previous, setPrevious] = useState(0);
  useEffect(() => {
    // After `current` change, set a 300ms timer making `previous = current` so the previous card will be removed.
    const timer = setTimeout(() => {
      setPrevious(current);
    }, 300);

    return () => {
      // Before `current` change to another value, remove (possibly not triggered) timer, and make `previous = current`.
      clearTimeout(timer);
      setPrevious(current);
    };
  }, [current]);
  return (
    <div className="friends" lang="zh-cn">
      <div style={{ position: "relative" }}>
        <div className="friend-columns">
          {/* Big card showing current selected */}
          <div className="friend-card-outer">
            {[
              previous != current && (
                <FriendCard key={previous} data={friends[previous]} fadeout />
              ),
              <FriendCard key={current} data={friends[current]} />,
            ]}
          </div>

          <div className="friend-list">
            {friends.map((x, i) => (
              <div
                key={x.name}
                className={clsx("friend-item", {
                  current: i == current,
                })}
                onClick={() => setCurrent(i)}
              >
                <img src={x.pic} alt="user profile photo" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function FriendCard(props: { data: FriendData; fadeout?: boolean }) {
  const { data, fadeout = false } = props;
  return (
    <div className={clsx("friend-card", { fadeout })}>
      <div className="card">
        <div className="card__image">
          <img
            src={data.pic}
            alt="User profile photo"
            title="User profile photo"
          />
        </div>
        <div className="card__body">
          <h2>{data.name}</h2>
          <p>
            <big>{data.intro}</big>
          </p>
          <p>
            <small>Comment : {data.note}</small>
          </p>
          <p>
            <small>IP : {data.ip}</small>
          </p>
        </div>
        <div className="card__footer">
          <a href={data.url} className="button button--primary button--block">
            Visit
          </a>
        </div>
      </div>
    </div>
  );
}

export default About;